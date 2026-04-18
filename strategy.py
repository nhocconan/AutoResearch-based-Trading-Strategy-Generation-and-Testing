#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_Volume_Regime_v1
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout signals on 12h timeframe.
Go long when price breaks above daily R1 with volume > 1.5x average and market in trending regime (Choppiness Index < 38.2).
Go short when price breaks below daily S1 with volume > 1.5x average and market in trending regime.
Exit when price returns to daily pivot (H5/L5) or regime shifts to choppy (Choppiness Index > 61.8).
Uses 1w trend filter to avoid counter-trend trades in strong trends.
Designed for low frequency (15-25 trades/year) to minimize fee decay while capturing major moves.
Works in bull markets via R1 breakouts and in bear via S1 breakdowns.
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
    
    # Get 1d data for Camarilla pivots and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    # H5 = Pivot + (H - L) * 1.1 / 2
    # L5 = Pivot - (H - L) * 1.1 / 2
    pivot_1d = np.full_like(high_1d, np.nan)
    r1_1d = np.full_like(high_1d, np.nan)
    s1_1d = np.full_like(low_1d, np.nan)
    h5_1d = np.full_like(high_1d, np.nan)
    l5_1d = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        hlc = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
        rng = high_1d[i-1] - low_1d[i-1]
        pivot_1d[i] = hlc
        r1_1d[i] = hlc + (rng * 1.1 / 12)
        s1_1d[i] = hlc - (rng * 1.1 / 12)
        h5_1d[i] = hlc + (rng * 1.1 / 2)
        l5_1d[i] = hlc - (rng * 1.1 / 2)
    
    # Calculate Choppiness Index (14-period) for regime detection
    # CHOP = 100 * log10(sum(ATR) / (max(HH) - min(LL))) / log10(n)
    atr_1d = np.full_like(high_1d, np.nan)
    tr_1d = np.full_like(high_1d, np.nan)
    
    # True Range
    for i in range(1, len(high_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        tr_1d[i] = tr
    
    # ATR (14-period Wilder smoothing)
    atr_period = 14
    if len(tr_1d) >= atr_period + 1:
        atr_1d[atr_period] = np.mean(tr_1d[1:atr_period+1])
        for i in range(atr_period + 1, len(tr_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr_1d[i]) / atr_period
    
    # Choppiness Index
    chop_1d = np.full_like(high_1d, np.nan)
    chop_period = 14
    if len(atr_1d) >= chop_period:
        for i in range(chop_period, len(high_1d)):
            sum_atr = np.sum(atr_1d[i-chop_period+1:i+1])
            max_hh = np.max(high_1d[i-chop_period+1:i+1])
            min_ll = np.min(low_1d[i-chop_period+1:i+1])
            if max_hh != min_ll:
                chop_1d[i] = 100 * np.log10(sum_atr / (max_hh - min_ll)) / np.log10(chop_period)
            else:
                chop_1d[i] = 50  # neutral when no range
    
    # Calculate 1w EMA34 for trend filter
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).values
    
    # Align all 1d indicators to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h5_1d_aligned = align_htf_to_ltf(prices, df_1d, h5_1d)
    l5_1d_aligned = align_htf_to_ltf(prices, df_1d, l5_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align 1w EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1  # need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(h5_1d_aligned[i]) or 
            np.isnan(l5_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: only trade in direction of 1w EMA34
        # For long: price > EMA34, for short: price < EMA34
        if not np.isnan(ema_1w_aligned[i]):
            long_filter = close[i] > ema_1w_aligned[i]
            short_filter = close[i] < ema_1w_aligned[i]
        else:
            long_filter = True
            short_filter = True
        
        if position == 0:
            # Long: price breaks above R1 + volume + chop < 38.2 (trending) + trend filter
            if (close[i] > r1_1d_aligned[i] and vol_confirm and 
                chop_1d_aligned[i] < 38.2 and long_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + chop < 38.2 (trending) + trend filter
            elif (close[i] < s1_1d_aligned[i] and vol_confirm and 
                  chop_1d_aligned[i] < 38.2 and short_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to H5 OR chop > 61.8 (choppy) OR reverse signal
            if (close[i] < h5_1d_aligned[i] or chop_1d_aligned[i] > 61.8 or
                close[i] < s1_1d_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to L5 OR chop > 61.8 (choppy) OR reverse signal
            if (close[i] > l5_1d_aligned[i] or chop_1d_aligned[i] > 61.8 or
                close[i] > r1_1d_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0