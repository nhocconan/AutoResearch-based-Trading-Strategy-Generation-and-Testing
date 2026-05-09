#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with daily trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; combined with daily trend
# and volume spikes, it captures mean-reversion entries in trending markets.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
# Target: 50-150 trades over 4 years with disciplined risk control.

name = "6h_WilliamsR_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on daily: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period highest high and lowest low
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    rr_diff = highest_high - lowest_low
    rr_diff[rr_diff == 0] = 1e-10
    
    williams_r = (highest_high - close_1d) / rr_diff * -100
    
    # Align Williams %R to 6h
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (24-period on 6h = 4 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema34_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in uptrend (price > EMA34) with volume spike
            if williams_r_6h[i] < -80 and close[i] > ema34_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in downtrend (price < EMA34) with volume spike
            elif williams_r_6h[i] > -20 and close[i] < ema34_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum fading) OR trend turns down
            if williams_r_6h[i] > -50 or close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -50 (momentum fading) OR trend turns up
            if williams_r_6h[i] < -50 or close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals