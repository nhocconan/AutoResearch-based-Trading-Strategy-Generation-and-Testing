#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "ETH APO Strategy"
timeframe = "1h"
leverage = 1

def calculate_ema(src, length):
    alpha = 2.0 / (length + 1)
    result = np.zeros_like(src)
    result[0] = src[0]
    for i in range(1, len(src)):
        result[i] = alpha * src[i] + (1 - alpha) * result[i-1]
    return result

def calculate_rma(src, length):
    alpha = 1.0 / length
    result = np.zeros_like(src)
    result[0] = src[0]
    for i in range(1, len(src)):
        result[i] = alpha * src[i] + (1 - alpha) * result[i-1]
    return result

def calculate_rsi(close, length=14):
    diff = np.zeros_like(close)
    diff[1:] = close[1:] - close[:-1]
    gain = np.maximum(diff, 0)
    loss = np.maximum(-diff, 0)
    avg_gain = calculate_rma(gain, length)
    avg_loss = calculate_rma(loss, length)
    rsi = np.zeros_like(close)
    mask_loss_zero = avg_loss == 0
    mask_gain_zero = avg_gain == 0
    mask_normal = ~mask_loss_zero & ~mask_gain_zero
    rsi[mask_loss_zero] = 100
    rsi[mask_gain_zero] = 0
    rsi[mask_normal] = 100 - (100 / (1 + avg_gain[mask_normal] / avg_loss[mask_normal]))
    return rsi

def calculate_sma(src, length):
    result = np.zeros_like(src)
    for i in range(length - 1, len(src)):
        result[i] = np.mean(src[i - length + 1:i + 1])
    return result

def generate_signals(prices):
    close = prices['close'].values
    n = len(close)
    if n == 0:
        return np.array([], dtype=int)
    
    ema_short = calculate_ema(close, 9)
    ema_long = calculate_ema(close, 19)
    apo = ema_short - ema_long
    
    rsi = calculate_rsi(close, 14)
    rsi_ma = calculate_sma(rsi, 5)
    
    apo = np.nan_to_num(apo, nan=0.0)
    rsi_ma = np.nan_to_num(rsi_ma, nan=0.0)
    
    signals = np.zeros(n, dtype=int)
    position = 0
    
    for i in range(n - 1):
        entry_trigger = False
        exit_trigger = False
        
        if i >= 1:
            apo_cross_over_neg15 = (apo[i] > -15) and (apo[i-1] <= -15)
            apo_cross_over_14 = (apo[i] > 14) and (apo[i-1] <= 14)
            
            rsi_cross_under_70 = (rsi_ma[i] < 70) and (rsi_ma[i-1] >= 70)
            rsi_cross_under_60 = (rsi_ma[i] < 60) and (rsi_ma[i-1] >= 60)
            rsi_cross_under_25 = (rsi_ma[i] < 25) and (rsi_ma[i-1] >= 25)
            
            rsi_ma_cond = False
            if i >= 2:
                rsi_ma_cond = rsi_ma[i] > rsi_ma[i-2]
            
            if apo_cross_over_neg15 and rsi_ma_cond:
                entry_trigger = True
            
            if apo_cross_over_14 or rsi_cross_under_70 or rsi_cross_under_60 or rsi_cross_under_25:
                exit_trigger = True
        
        if position == 1 and exit_trigger:
            position = 0
        elif position == 0 and entry_trigger:
            position = 1
        
        signals[i+1] = position
    
    return signals
