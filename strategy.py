#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d volume confirmation and 1w EMA trend filter
# Targets overbought/oversold reversals in trending markets with volume confirmation
# Designed for low trade frequency (target 20-40/year) by requiring multiple confluence
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R (14-period) on 4h
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low + 1e-10)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Williams %R thresholds: oversold < -80, overbought > -20
        # Long entry: Williams %R oversold + uptrend + volume spike
        if (williams_r_aligned[i] < -80 and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought + downtrend + volume spike
        elif (williams_r_aligned[i] > -20 and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R returns to neutral zone (-50) or trend reversal
        elif position == 1 and (williams_r_aligned[i] > -50 or 
                                close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] < -50 or 
                                 close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_1dVolume_1wEMA_Reversal"
timeframe = "4h"
leverage = 1.0