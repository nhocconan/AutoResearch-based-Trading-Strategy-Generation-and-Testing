#!/usr/bin/env python3
# 1d_keltner_channel_1w_trend_volume
# Hypothesis: Keltner Channel breakout on daily timeframe with weekly EMA trend filter and volume confirmation.
# Long when price breaks above upper Keltner band with uptrend (price > weekly EMA50) and volume > 1.5x average.
# Short when price breaks below lower Keltner band with downtrend (price < weekly EMA50) and volume > 1.5x average.
# Exit when price crosses back to the EMA20 (middle band).
# Designed to capture strong breakouts with trend alignment in both bull and bear markets.
# Target: 30-100 total trades over 4 years (~7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_channel_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Keltner Channel on daily data (20-period EMA, 2x ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 (middle band)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 (middle band)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout entries: Keltner upper breakout (long) and lower breakdown (short)
            if (close[i] > upper_keltner[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < lower_keltner[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals