#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme readings with volume spike and 1d EMA200 trend filter.
Long when Williams %R < -80 (oversold) AND volume > 1.8x average AND price > 1d EMA200 (uptrend).
Short when Williams %R > -20 (overbought) AND volume > 1.8x average AND price < 1d EMA200 (downtrend).
Exit when Williams %R returns to -50 level OR volume drops below average.
Uses 6h for price/momentum/volume, 1d for EMA200 trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R identifies exhaustion points,
volume confirmation reduces fakeouts, daily EMA200 ensures we trade with the higher timeframe trend.
Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
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
    
    # Calculate Williams %R on 6h timeframe (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.inf)
    
    # Calculate volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 6h Williams %R, volume MA, and 1d EMA200 to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # same timeframe, no alignment needed but using helper for consistency
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
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
            # Long: Williams %R < -80 (oversold) AND volume > 1.8x avg AND price > 1d EMA200 (uptrend)
            if wr < -80 and vol > 1.8 * vol_ma and price > ema_200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume > 1.8x avg AND price < 1d EMA200 (downtrend)
            elif wr > -20 and vol > 1.8 * vol_ma and price < ema_200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (return to neutral) OR volume < average (losing momentum)
            if wr > -50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (return to neutral) OR volume < average (losing momentum)
            if wr < -50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_Volume_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0