#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe using 1d Williams %R (Williams Percent Range) as mean-reversion signal
# Williams %R > -20 = overbought, < -80 = oversold. Combined with 1d EMA trend filter and volume spike.
# Works in both bull/bear: mean reversion in ranges, trend-following in strong trends via EMA filter.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize fee churn.
name = "6h_WilliamsR_MeanRev_1dTrend_VolumeSpike"
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
    
    # Get 1d data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where(highest_high == lowest_low, -50, williams_r)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Williams %R and EMA50 to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection (24-period for 6h = 4 days of volume)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for Williams %R (14) + EMA (50) + volume (24)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema50_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Oversold condition (Williams %R < -80) with uptrend bias and volume spike
            if williams_r_6h[i] < -80 and close[i] > ema50_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Overbought condition (Williams %R > -20) with downtrend bias and volume spike
            elif williams_r_6h[i] > -20 and close[i] < ema50_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral territory (> -50) OR trend turns down
            if williams_r_6h[i] > -50 or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral territory (< -50) OR trend turns up
            if williams_r_6h[i] < -50 or close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals