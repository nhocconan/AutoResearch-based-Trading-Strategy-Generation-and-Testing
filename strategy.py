#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in 1d uptrend with volume spike (>1.5x 20-period volume MA).
# Short when Williams %R > -20 (overbought) in 1d downtrend with volume spike.
# Williams %R identifies exhaustion points; 1d EMA50 ensures higher timeframe alignment to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 4h timeframe to achieve 75-200 total trades over 4 years.

name = "4h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 4h data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_4h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_4h['low']).rolling(window=14, min_periods=14).min().values
    close_4h = df_4h['close'].values
    
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
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
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        williams_r_val = williams_r_aligned[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND 1d uptrend AND volume spike
            if williams_r_val < -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND 1d downtrend AND volume spike
            elif williams_r_val > -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (exiting oversold) OR trend changes
            if williams_r_val > -50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (exiting overbought) OR trend changes
            if williams_r_val < -50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals