#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA50 trend filter and volume spike confirmation.
# Williams %R measures momentum reversal points: oversold > -20, overbought < -80.
# 1w EMA50 filter ensures we trade only in the direction of the weekly trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Replace division by zero or near-zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (> -20) AND uptrend AND volume spike
            if williams_r[i] > -20 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (< -80) AND downtrend AND volume spike
            elif williams_r[i] < -80 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R becomes overbought OR trend reverses
            if williams_r[i] < -80 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R becomes oversold OR trend reverses
            if williams_r[i] > -20 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals