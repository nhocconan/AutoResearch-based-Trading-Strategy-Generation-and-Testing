#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-period Donchian channels (weekly)
    if len(high_1w) < 20:
        return np.zeros(n)
    
    # Upper band: highest high over last 20 weeks
    upper_20 = np.full_like(high_1w, np.nan)
    for i in range(19, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-19:i+1])
    
    # Lower band: lowest low over last 20 weeks
    lower_20 = np.full_like(low_1w, np.nan)
    for i in range(19, len(low_1w)):
        lower_20[i] = np.min(low_1w[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Calculate 40-week EMA for trend filter (weekly)
    if len(close_1w) < 40:
        return np.zeros(n)
    
    ema40_1w = np.full_like(close_1w, np.nan)
    ema40_1w[0] = close_1w[0]
    alpha = 2 / (40 + 1)
    for i in range(1, len(close_1w)):
        ema40_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema40_1w[i-1]
    
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate 14-week RSI for momentum (weekly)
    if len(close_1w) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1w, np.nan)
    avg_loss = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1w, np.nan)
    rsi14 = np.full_like(close_1w, np.nan)
    for i in range(13, len(close_1w)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi14[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_aligned = align_htf_to_ltf(prices, df_1w, rsi14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(40, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or 
            np.isnan(rsi14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current weekly volume vs 20-period average
        vol_ma_20 = np.full_like(volume_1w, np.nan)
        for j in range(19, len(volume_1w)):
            vol_ma_20[j] = np.mean(volume_1w[j-19:j+1])
        
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
        
        if np.isnan(vol_ma_20_aligned[i]) or vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_1w[-1] / vol_ma_20_aligned[i] if len(volume_1w) > 0 else 0
        
        if position == 0:
            # Long: Price breaks above upper Donchian + price above EMA40 + RSI > 55 + volume surge
            if (close[i] > upper_20_aligned[i] and
                close[i] > ema40_1w_aligned[i] and
                rsi14_aligned[i] > 55 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian + price below EMA40 + RSI < 45 + volume surge
            elif (close[i] < lower_20_aligned[i] and
                  close[i] < ema40_1w_aligned[i] and
                  rsi14_aligned[i] < 45 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below lower Donchian OR RSI < 40
            if (close[i] < lower_20_aligned[i] or 
                rsi14_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above upper Donchian OR RSI > 60
            if (close[i] > upper_20_aligned[i] or 
                rsi14_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_EMA40_RSI14_Volume_v1"
timeframe = "1d"
leverage = 1.0