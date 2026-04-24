#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation.
- Williams %R(14) < -80 = oversold, > -20 = overbought
- Enter long on Williams %R crossing above -80 from below with volume spike and price > 1d EMA50
- Enter short on Williams %R crossing below -20 from above with volume spike and price < 1d EMA50
- Uses 1d EMA50 for trend alignment to avoid counter-trend trades in strong markets
- Volume spike (>2.0x 20-period average) confirms conviction
- Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear via trend filter and mean-reversion logic at extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) calculation
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (exiting oversold)
            # with volume spike and price above 1d EMA50 (uptrend bias)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_spike[i] and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (exiting overbought)
            # with volume spike and price below 1d EMA50 (downtrend bias)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_spike[i] and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (entering overbought)
            # or price closes below 1d EMA50 (trend change)
            if williams_r[i] >= -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (entering oversold)
            # or price closes above 1d EMA50 (trend change)
            if williams_r[i] <= -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0