#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1w EMA50 trend filter and volume spike confirmation
- Williams %R(14) identifies overbought/oversold conditions on 6h chart
- 1w EMA50 establishes long-term trend direction (avoids counter-trend trades)
- Volume spike (2.0x 20-period MA) confirms institutional participation at extremes
- Mean reversion: long when %R < -80 in uptrend, short when %R > -20 in downtrend
- Exit when %R reverts to -50 level
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
- Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 6h
    def williams_r(high_arr, low_arr, close_arr, window=14):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        # Handle division by zero (when high == low)
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14 = williams_r(high, low, close, 14)
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(wr_14[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr_14[i]
        ema_trend = ema50_1w_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend alignment
            # Long: Williams %R oversold (< -80) + volume spike + price above 1w EMA50 (uptrend)
            if wr_val < -80.0 and vol > 2.0 * vol_ma and close[i] > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + volume spike + price below 1w EMA50 (downtrend)
            elif wr_val > -20.0 and vol > 2.0 * vol_ma and close[i] < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R reverts to -50 (mean reversion complete)
            if wr_val > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R reverts to -50 (mean reversion complete)
            if wr_val < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1wEMA50_VolumeSpike_MeanReversion"
timeframe = "6h"
leverage = 1.0