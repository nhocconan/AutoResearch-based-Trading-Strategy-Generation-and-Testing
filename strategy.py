#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA50 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (> -20 or < -80) signal reversals.
# 1w EMA50 filter ensures trades align with the long-term trend, reducing counter-trend whipsaws.
# Volume spike confirms conviction in the reversal move.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
name = "6h_WilliamsR_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
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
            # Long: Williams %R oversold (< -80) AND uptrend AND volume spike
            if williams_r[i] < -80 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume spike
            elif williams_r[i] > -20 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 OR trend reverses
            if williams_r[i] > -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 OR trend reverses
            if williams_r[i] < -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals