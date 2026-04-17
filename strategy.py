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
    
    # === 1d Weekly Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period highest high and lowest low
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    period = 20
    for i in range(len(high_1d)):
        if i >= period - 1:
            highest_high[i] = np.max(high_1d[i-(period-1):i+1])
            lowest_low[i] = np.min(low_1d[i-(period-1):i+1])
        else:
            highest_high[i] = np.max(high_1d[0:i+1]) if i >= 0 else high_1d[0]
            lowest_low[i] = np.min(low_1d[0:i+1]) if i >= 0 else low_1d[0]
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    rsi_period = 14
    for i in range(len(gain)):
        if i < rsi_period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # === 1d EMA(20) for trend filter ===
    ema_20 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20[19] = np.mean(close_1d[:20])  # seed
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema_20[i] = alpha * close_1d[i] + (1 - alpha) * ema_20[i-1]
    else:
        for i in range(len(close_1d)):
            ema_20[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to 1d timeframe (same timeframe, no alignment needed) ===
    # Since we're using 1d data on 1d timeframe, we can use the values directly
    highest_high_aligned = highest_high
    lowest_low_aligned = lowest_low
    rsi_aligned = rsi
    ema_20_aligned = ema_20
    
    # === Volume confirmation (using 1d volume) ===
    vol_ma_20 = np.full_like(df_1d['volume'].values, np.nan)
    vol_1d = df_1d['volume'].values
    for i in range(len(vol_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(vol_1d[0:i+1]) if i >= 0 else vol_1d[0]
    
    # Align volume MA to 1d timeframe
    vol_ma_20_aligned = vol_ma_20
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_confirm = vol_1d > vol_ma_20_aligned * 1.3
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Close breaks above 20-day high AND RSI > 50 AND volume confirmation
            if (close[i] > highest_high_aligned[i] and 
                rsi_aligned[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below 20-day low AND RSI < 50 AND volume confirmation
            elif (close[i] < lowest_low_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Close crosses below 20-day EMA OR RSI < 40
            if (close[i] < ema_20_aligned[i]) or (rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above 20-day EMA OR RSI > 60
            if (close[i] > ema_20_aligned[i]) or (rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_RSI50_EMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0