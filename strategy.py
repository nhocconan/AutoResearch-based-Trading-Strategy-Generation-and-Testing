#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; EMA34 determines trend direction for bias
# Volume spike confirms conviction; discrete sizing 0.25 limits fee drag; target 50-150 trades over 4 years
# Works in bull/bear: mean reversion in range, trend-filtered continuation in strong moves

name = "6h_WilliamsR_MeanRev_12hEMA34_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34 trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume average (20-period)
    volume_12h_series = pd.Series(volume_12h)
    avg_volume_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (1.5 * avg_volume_12h)  # 50% above average volume
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align HTF indicators to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend (close > EMA34) AND volume spike
            if williams_r[i] < -80 and close[i] > ema34_12h_aligned[i] and volume_spike_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend (close < EMA34) AND volume spike
            elif williams_r[i] > -20 and close[i] < ema34_12h_aligned[i] and volume_spike_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend breaks
            if williams_r[i] > -50 or close[i] <= ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend breaks
            if williams_r[i] < -50 or close[i] >= ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals