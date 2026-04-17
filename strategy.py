#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal + 1d EMA50 trend filter + volume spike
- Williams %R(14) identifies overbought (> -20) and oversold (< -80) conditions on 6h
- Trend filter: 1d EMA50 slope confirms bias (long only when EMA50 rising, short only when falling)
- Volume confirmation: 2.0x 20-period MA on 6h filters weak signals
- Fixed position size 0.25 to manage drawdown and reduce fee churn
- Works in bull markets (buying oversold in uptrend) and bear markets (selling overbought in downtrend)
- Williams %R is mean-reverting but requires trend alignment to avoid whipsaw in strong trends
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
    
    # Get 1d data for EMA50 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.gradient(ema50_1d)  # slope of EMA50
    
    # Williams %R(14) on 6h (primary timeframe)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 slope to 6h timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        ema_slope = ema50_slope_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        
        if position == 0:
            # Look for extreme Williams %R reversals with volume and trend confirmation
            # Long: Williams %R < -80 (oversold) + volume spike + EMA50 rising
            if wr < -80 and vol > 2.0 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + EMA50 falling
            elif wr > -20 and vol > 2.0 * vol_ma and ema_slope < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R returns above -50 (mean reversion) or trend fails
            if wr > -50 or ema_slope <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R returns below -50 (mean reversion) or trend fails
            if wr < -50 or ema_slope >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0