#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_ChopFilter
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 12h EMA50 trend filter and choppiness regime (CHOP < 50) captures strong directional moves while avoiding choppy markets. The choppiness filter reduces false breakouts in ranging conditions, improving win rate and reducing trade frequency. Designed for 20-40 trades/year to minimize fee drag. Works in both bull and bear markets via trend filter and regime adaptation.
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
    
    # Get 12h data for HTF trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (R1, S1)
    prev_close = np.concatenate([[np.nan], close_12h[:-1]])
    prev_high = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low = np.concatenate([[np.nan], low_12h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Calculate choppiness index on 4h (14-period)
    # CHOP = 100 * log10(sum(ATR(1) over 14) / log10(range(14))) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / (max_high_14 - min_low_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high_14 - min_low_14) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 14)  # EMA50, CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        chop_val = chop[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Choppiness filter: only trade when market is not too choppy (CHOP < 50)
        not_choppy = chop_val < 50.0
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend and non-choppy market
            # Long: price breaks above R1 with uptrend (close > EMA50) and not choppy
            long_signal = (high_val > r1_val) and (close_val > ema_val) and not_choppy
            # Short: price breaks below S1 with downtrend (close < EMA50) and not choppy
            short_signal = (low_val < s1_val) and (close_val < ema_val) and not_choppy
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0