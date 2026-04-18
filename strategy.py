#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, price breaks above Camarilla R1 or below S1 from prior 12h bar with volume > 1.5x 20-period average and choppy market filter (Choppiness Index > 61.8 for mean reversion). Long at R1 breakout, short at S1 breakout. Uses 1w EMA34 for trend filter: only long when price > weekly EMA34, short when price < weekly EMA34. Designed for low trade frequency (~15-25/year) to avoid fee drag, works in bull via trend-aligned breaks and in bear via mean reversion in chop.
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_width = 1.1 * (high_12h - low_12h) / 12.0
    r1_12h = close_12h + camarilla_width
    s1_12h = close_12h - camarilla_width
    
    # Align Camarilla levels to 12h timeframe (already aligned, but use for safety)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Choppiness Index on 12h (for regime filter)
    def choppiness_index(high, low, close, period=14):
        atr = np.full_like(high, np.nan)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period if not np.isnan(atr[i-1]) else tr[i]
        # For first period values, use simple average
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
        
        hhll = np.zeros_like(high)
        llh = np.zeros_like(low)
        for i in range(len(high)):
            if i == 0:
                hhll[i] = high[i]
                llh[i] = low[i]
            else:
                hhll[i] = max(high[i], hhll[i-1])
                llh[i] = min(low[i], llh[i-1])
        
        chop = np.full_like(high, np.nan)
        for i in range(period, len(high)):
            if hhll[i] > llh[i]:
                sum_atr = np.sum(atr[i-period+1:i+1]) if i >= period else np.sum(atr[:i+1])
                chop[i] = 100 * np.log10(sum_atr / (hhll[i] - llh[i])) / np.log10(period)
        return chop
    
    chop_12h = choppiness_index(high_12h, low_12h, close_12h, 14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma_12h = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma_12h[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1  # Need at least one 12h bar and vol MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_12h[i]
        
        # Chop regime: only trade in choppy markets (Choppiness > 61.8) for mean reversion
        in_chop = chop_12h_aligned[i] > 61.8
        
        if position == 0 and vol_confirm and in_chop:
            # Long: price breaks above R1 and above weekly EMA (uptrend filter)
            if close[i] > r1_12h_aligned[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 and below weekly EMA (downtrend filter)
            elif close[i] < s1_12h_aligned[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR chop breaks down (trend emerging)
            if close[i] < s1_12h_aligned[i] or chop_12h_aligned[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR chop breaks down (trend emerging)
            if close[i] > r1_12h_aligned[i] or chop_12h_aligned[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0
EOF