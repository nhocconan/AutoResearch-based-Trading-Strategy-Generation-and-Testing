#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 12h volume spike and 1d EMA200 trend filter.
Long when Williams %R < -80 (oversold) AND volume > 2x 12h average AND price > 1d EMA200 (uptrend).
Short when Williams %R > -20 (overbought) AND volume > 2x 12h average AND price < 1d EMA200 (downtrend).
Exit when Williams %R reverts to -50 (mean reversion) OR volume < 1.2x average (momentum fade).
Uses 6h for Williams %R calculation, 12h for volume confirmation, and 1d for EMA200 trend filter.
Targets 50-150 total trades over 4 years (12-37/year). Williams %R captures reversals at extremes,
volume spike confirms institutional interest, EMA200 filter ensures alignment with higher timeframe trend.
Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R on 6h timeframe (14-period)
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close_6h) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.inf))
    
    # Get 12h data for volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_200 = ema_200_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume > 2x average AND price > EMA200 (uptrend)
            if wr < -80 and vol > 2.0 * vol_ma and price > ema_200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume > 2x average AND price < EMA200 (downtrend)
            elif wr > -20 and vol > 2.0 * vol_ma and price < ema_200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (mean reversion) OR volume < 1.2x average (momentum fade)
            if wr > -50 or vol < 1.2 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (mean reversion) OR volume < 1.2x average (momentum fade)
            if wr < -50 or vol < 1.2 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_VolumeSpike_EMA200_Filter"
timeframe = "6h"
leverage = 1.0