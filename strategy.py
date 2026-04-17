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
    
    # === 1w 20-period Donchian channels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Upper band: highest high of last 20 weeks
    upper_20 = np.full_like(high_1w, np.nan)
    for i in range(len(high_1w)):
        if i >= 19:
            upper_20[i] = np.max(high_1w[i-19:i+1])
        elif i > 0:
            upper_20[i] = np.max(high_1w[0:i+1])
        else:
            upper_20[i] = high_1w[0]
    
    # Lower band: lowest low of last 20 weeks
    lower_20 = np.full_like(low_1w, np.nan)
    for i in range(len(low_1w)):
        if i >= 19:
            lower_20[i] = np.min(low_1w[i-19:i+1])
        elif i > 0:
            lower_20[i] = np.min(low_1w[0:i+1])
        else:
            lower_20[i] = low_1w[0]
    
    # === 1d 14-period RSI for momentum filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    period = 14
    
    for i in range(len(close_1d)):
        if i < period:
            if i > 0:
                avg_gain[i] = np.mean(gain[0:i+1])
                avg_loss[i] = np.mean(loss[0:i+1])
            else:
                avg_gain[i] = gain[0]
                avg_loss[i] = loss[0]
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)
    rsi_1d = np.where(avg_gain == 0, 0, rsi_1d)
    
    # === 1d 20-period EMA for trend filter ===
    ema_20 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20[19] = np.mean(close_1d[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema_20[i] = alpha * close_1d[i] + (1 - alpha) * ema_20[i-1]
    else:
        for i in range(len(close_1d)):
            ema_20[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to 1d timeframe ===
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # === 1d Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly upper band AND RSI > 50 AND price above daily EMA20
            if (close[i] > upper_20_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                close[i] > ema_20_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly lower band AND RSI < 50 AND price below daily EMA20
            elif (close[i] < lower_20_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  close[i] < ema_20_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below weekly lower band OR RSI < 30
            if (close[i] < lower_20_aligned[i] or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly upper band OR RSI > 70
            if (close[i] > upper_20_aligned[i] or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_RSI50_EMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0