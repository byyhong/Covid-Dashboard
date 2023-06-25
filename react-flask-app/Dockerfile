FROM node:20-alpine
WORKDIR /react-flask-app
COPY package*.json ./
COPY . .
RUN npm ci
EXPOSE 3000
CMD ["npm","run","test"]