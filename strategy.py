#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(50)
    ema_50_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50_1d[i] = close_1d[i]
        elif i < 50:
            ema_50_1d[i] = np.mean(close_1d[:i+1])
        else:
            if np.isnan(ema_50_1d[i-1]):
                ema_50_1d[i] = np.mean(close_1d[i-49:i+1])
            else:
                ema_50_1d[i] = close_1d[i] * alpha + ema_50_1d[i-1] * (1 - alpha)
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(close_1d), np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14_1d = np.full(len(close_1d), np.nan)
    rsi_14_1d[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get weekly data for trend filter: EMA(34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_34_1w[i] = close_1w[i]
        elif i < 34:
            ema_34_1w[i] = np.mean(close_1w[:i+1])
        else:
            if np.isnan(ema_34_1w[i-1]):
                ema_34_1w[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_34_1w[i] = close_1w[i] * alpha_w + ema_34_1w[i-1] * (1 - alpha_w)
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: ATR > 0.02 * price (avoid choppy markets)
        vol_filter = atr_14_1d_aligned[i] > 0.02 * price
        
        if position == 0:
            # Long: Price > EMA50 + RSI < 40 + Weekly Uptrend
            if (price > ema_50_1d_aligned[i] and 
                rsi_14_1d_aligned[i] < 40 and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price < EMA50 + RSI > 60 + Weekly Downtrend
            elif (price < ema_50_1d_aligned[i] and 
                  rsi_14_1d_aligned[i] > 60 and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price < EMA50 or RSI > 60 or Weekly trend turns down
            if (price < ema_50_1d_aligned[i] or 
                rsi_14_1d_aligned[i] > 60 or 
                ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > EMA50 or RSI < 40 or Weekly trend turns up
            if (price > ema_50_1d_aligned[i] or 
                rsi_14_1d_aligned[i] < 40 or 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA50_RSI14_WeeklyEMA34_v1"
timeframe = "1d"
leverage = 1.0