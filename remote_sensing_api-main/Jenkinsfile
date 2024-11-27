pipeline {
    agent any
    stages {
        stage('Setup Python Virtual ENV for dependencies') {
            steps {
                script {
                    // Specify the absolute path to the script in the correct directory
                    sh '''
                    chmod +x envsetup.sh
                    dos2unix envsetup.sh
                    ./envsetup.sh
                    '''
                }
            }
        }
        stage('Setup Gunicorn') {
            steps {
                script {
                    // Specify the absolute path to the script in the correct directory
                    sh '''
                    chmod +x gunicorn.sh
                    dos2unix gunicorn.sh
                    ./gunicorn.sh
                    '''
                }
            }
        }
  
     
    }
}