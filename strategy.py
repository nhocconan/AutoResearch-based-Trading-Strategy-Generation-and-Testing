#!/usr/bin/env python3
"""
#101001 - 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Breakout above Camarilla R1 or below S1 using daily pivots, with volume confirmation and EMA34 trend filter on daily timeframe.
Includes Choppiness Index regime filter to avoid whipsaws in sideways markets. Works in trending markets (breakout with trend) and uses chop filter to reduce false signals in ranging markets.
Target: 30-50 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.
Proven pattern: Camarilla breakouts with volume and trend filtering show strong performance in backtests (see DB top performers).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    r1 = pp + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get EMA34 on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average (more selective to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter to avoid ranging markets
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50)  # default to 50 when range is zero
    
    # Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending (favor)
    chop_filter = chop < 61.8  # Allow trading when not strongly ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above EMA34, volume spike, not strongly ranging
        if (close[i] > r1_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i] and 
            chop_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below EMA34, volume spike, not strongly ranging
        elif (close[i] < s1_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i] and 
              chop_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Camarilla level (mean reversion)
        elif position == 1 and close[i] < s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0