#!/usr/bin/env python3
"""
#100989 - 4h_Donchian20_Breakout_VolumeTrend_Filter
Hypothesis: Breakout above Donchian(20) high or below Donchian(20) low with volume confirmation and EMA50 trend filter on 4h timeframe.
Includes Choppiness Index regime filter to avoid whipsaws in sideways markets. Works in trending markets (breakout with trend) and uses chop filter to reduce false signals in ranging markets.
Target: 30-50 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.
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
    
    # Get 4h data for Donchian calculation (to ensure proper 4h alignment)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels on 4h data
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 15m timeframe (primary timeframe is 4h, but we need to check)
    # Actually, since primary is 4h, we can use the values directly but need to align for safety
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above EMA50, volume spike, not strongly ranging
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i] and 
            chop_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below EMA50, volume spike, not strongly ranging
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i] and 
              chop_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian band (mean reversion)
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
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

name = "4h_Donchian20_Breakout_VolumeTrend_Filter"
timeframe = "4h"
leverage = 1.0