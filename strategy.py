#!/usr/bin/env python3
# 12h_adx_ma_crossover_volume
# Hypothesis: Moving average crossover (EMA21/EMA50) on 12h filtered by ADX trend strength (ADX>25) and volume confirmation (volume>1.5x average). Long when EMA21 crosses above EMA50 with strong trend and volume; short when EMA21 crosses below EMA50 with strong trend and volume. Designed to capture trending moves while avoiding choppy markets. Target: 20-40 trades/year (~80-160 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adx_ma_crossover_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period]) if not np.isnan(data[:period]).any() else 0
        for i in range(period, len(data)):
            result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    # Calculate smoothed TR, +DM, -DM
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Handle division by zero and NaN
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.where(np.isnan(adx), 0, adx)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA21 and EMA50 on 12h data
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA21 crosses below EMA50 OR trend weakens (ADX<20)
            if (ema_21[i] < ema_50[i]) or (adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA21 crosses above EMA50 OR trend weakens (ADX<20)
            if (ema_21[i] > ema_50[i]) or (adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: EMA21 crosses above EMA50 with strong trend and volume
            if (ema_21[i] > ema_50[i]) and (adx_aligned[i] > 25) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA21 crosses below EMA50 with strong trend and volume
            elif (ema_21[i] < ema_50[i]) and (adx_aligned[i] > 25) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals