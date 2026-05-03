#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in 1d uptrend with volume spike (>1.5x 20-period volume MA).
# Short when Williams %R > -20 (overbought) in 1d downtrend with volume spike.
# Williams %R identifies extreme price levels for mean reversion. 1d EMA50 ensures higher timeframe alignment,
# avoiding counter-trend trades during bear market rallies. Volume spike confirms institutional participation.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with discrete sizing.

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 14:  # Need at least 14 periods for Williams %R
        return np.zeros(n)
    
    # Calculate Williams %R on 6h data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 14-period highest high and lowest low
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    williams_r = ((highest_high - close_6h) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align Williams %R to lower timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        wr = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        trend_up = close[i] > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close[i] < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND 1d uptrend AND volume spike
            if wr < -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND 1d downtrend AND volume spike
            elif wr > -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or 1d trend changes
            if wr > -50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or 1d trend changes
            if wr < -50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals