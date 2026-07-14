function qs(sel, el=document){return el.querySelector(sel)}
function qsa(sel, el=document){return el.querySelectorAll(sel)}

const authArea = qs('#auth')
const loginModal = qs('#login-modal')
const showLoginBtn = qs('#show-login')
const cancelLoginBtn = qs('#cancel-login')
const loginForm = qs('#login-form')
const loginError = qs('#login-error')
const signupModal = qs('#signup-modal')
const signupForm = qs('#signup-form')
const signupError = qs('#signup-error')
const cancelSignupBtn = qs('#cancel-signup')

function showModal(){ loginModal.classList.remove('hidden') }
function hideModal(){ loginModal.classList.add('hidden'); loginError.textContent = '' }

function showSignup(){ signupModal.classList.remove('hidden') }
function hideSignup(){ signupModal.classList.add('hidden'); signupError.textContent = '' }

async function setAuthenticatedUI(){
  const token = localStorage.getItem('access_token')
  if(!token){
    authArea.innerHTML = `<button id="show-login" class="btn">Sign In</button>`
    qs('#show-login').addEventListener('click', showModal)
    return
  }
  try{
    const res = await fetch('/api/auth/me', { headers: { 'Authorization': 'Bearer '+token } })
    if(!res.ok) throw new Error('Not authenticated')
    const user = await res.json()
    authArea.innerHTML = `<span class="welcome">Hi, ${user.name}</span> <button id="logout" class="btn">Log out</button>`
    qs('#logout').addEventListener('click', ()=>{ localStorage.removeItem('access_token'); localStorage.removeItem('refresh_token'); setAuthenticatedUI() })
  }catch(e){
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    authArea.innerHTML = `<button id="show-login" class="btn">Sign In</button>`
    qs('#show-login').addEventListener('click', showModal)
  }
}

showLoginBtn && showLoginBtn.addEventListener('click', showModal)
cancelLoginBtn && cancelLoginBtn.addEventListener('click', hideModal)
cancelSignupBtn && cancelSignupBtn.addEventListener('click', hideSignup)

loginForm && loginForm.addEventListener('submit', async (ev)=>{
  ev.preventDefault()
  loginError.textContent = ''
  const form = new FormData(loginForm)
  const email = form.get('email')
  const password = form.get('password')
  try{
    const body = new URLSearchParams({ username: email, password })
    const res = await fetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body })
    if(!res.ok){
      const err = await res.json().catch(()=>({detail:'Authentication failed'}))
      loginError.textContent = err.detail || 'Sign in failed'
      return
    }
    const data = await res.json()
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    hideModal()
    await setAuthenticatedUI()
  }catch(err){
    loginError.textContent = 'Network error'
  }
})

// Signup handling
signupForm && signupForm.addEventListener('submit', async (ev)=>{
  ev.preventDefault()
  signupError.textContent = ''
  const form = new FormData(signupForm)
  const payload = {
    name: form.get('name'),
    email: form.get('email'),
    password: form.get('password'),
    age: Number(form.get('age')),
    experience_level: form.get('experience_level') || 'beginner',
  }
  try{
    const res = await fetch('/api/auth/signup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    if(!res.ok){
      const err = await res.json().catch(()=>({detail:'Signup failed'}))
      signupError.textContent = err.detail || 'Signup failed'
      return
    }
    // auto-login after signup
    const loginBody = new URLSearchParams({ username: payload.email, password: payload.password })
    const loginRes = await fetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: loginBody })
    if(!loginRes.ok){ signupError.textContent='Created but sign-in failed'; return }
    const data = await loginRes.json()
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    hideSignup()
    await setAuthenticatedUI()
  }catch(err){ signupError.textContent = 'Network error' }
})

// wire signup link from login modal
const showSignupFromLogin = qs('#show-signup')
if(showSignupFromLogin){ showSignupFromLogin.addEventListener('click', ()=>{ hideModal(); showSignup() }) }

// initialize
setAuthenticatedUI()
