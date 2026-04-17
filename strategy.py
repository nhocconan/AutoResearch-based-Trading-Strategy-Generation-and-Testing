#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1w EMA trend filter.
Long when price breaks above R1 AND volume > 1.3x average AND close > 1w EMA34.
Short when price breaks below S1 AND volume > 1.3x average AND close < 1w EMA34.
Exit when price reverts to the 1d close (mean reversion) or volume drops below average.
Uses 12h for entry timing, 1w for trend filter, and 1d for Camarilla calculation.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide institutional
reference points, volume confirms breakout strength, weekly EMA ensures we trade with the
higher timeframe trend. Works in bull markets (buy R1 breakouts in uptrend) and bear markets
(sell S1 breakdowns in downtrend).
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
    
    # Get 1d data for Camarilla calculation (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rang = high_1d - low_1d
    r1 = close_1d + 1.1 * rang / 12
    s1 = close_1d - 1.1 * rang / 12
    # Pivot point for exit reference
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume average (20-period) on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.3x avg AND close > 1w EMA34 (uptrend)
            if price > r1_val and vol > 1.3 * vol_ma and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.3x avg AND close < 1w EMA34 (downtrend)
            elif price < s1_val and vol > 1.3 * vol_ma and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < pivot point OR volume < average (loss of momentum)
            if price < pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > pivot point OR volume < average (loss of momentum)
            if price > pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wEMA34_Filter"
timeframe = "12h"
leverage = 1.0