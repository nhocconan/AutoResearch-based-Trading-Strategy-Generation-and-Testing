#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA50 trend filter and 6h volume confirmation
- Williams %R(14) on 1d: long when < -80 (oversold) and EMA50 rising, short when > -20 (overbought) and EMA50 falling
- Volume confirmation: 6h volume > 1.5x 20-period 6h MA to avoid false signals in low volatility
- Fixed position size 0.25 to limit fee churn and manage drawdown
- Designed to work in bull markets (buying oversold dips in uptrends) and bear markets (selling overbought rallies in downtrends)
- Uses 1d timeframe for Williams %R and EMA50 to reduce noise, 6h for entry timing and volume filter
- Target: 50-150 total trades over 4 years to avoid fee drag
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
    
    # Get 1d data for Williams %R and EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 and its slope
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.gradient(ema50_1d)  # slope of EMA50
    
    # Get 6h data for volume confirmation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    
    # Volume average (20-period) on 6h
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_slope = ema50_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend filter
            # Long: Williams %R < -80 (oversold) + volume spike + EMA50 rising
            if wr < -80 and vol > 1.5 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + EMA50 falling
            elif wr > -20 and vol > 1.5 * vol_ma and ema_slope < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R rises above -50 (mean reversion) or EMA50 turns down
            if wr > -50 or ema_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R falls below -50 (mean reversion) or EMA50 turns up
            if wr < -50 or ema_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeSpike_MeanRev"
timeframe = "6h"
leverage = 1.0