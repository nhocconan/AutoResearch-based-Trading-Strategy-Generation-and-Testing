# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Pivot Point reversal + volume confirmation + ATR volatility filter.
Pivot points act as strong support/resistance levels where price often reverses.
In ranging markets (common in 2025), reversals at pivot levels provide edge.
Volume confirmation ensures breakouts are genuine.
ATR filter avoids choppy, low-volatility environments where false signals occur.
Designed to work in both bull (buy at support) and bear (sell at resistance) markets.
Target: 15-25 trades/year to minimize fee drag.
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
    
    # Get daily data for pivot points (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - high_1d
    s1 = 2 * pivot - low_1d
    
    # Align pivot levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ATR(14) for volatility filter on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: compare current 12h volume to 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop) and high volatility (chaos)
        # Use 30-period ATR percentile to define normal volatility range
        if i >= 30:
            atr_slice = atr_14_1d_aligned[max(0, i-30):i+1]
            atr_20th = np.nanpercentile(atr_slice, 20)
            atr_80th = np.nanpercentile(atr_slice, 80)
            vol_filter = (atr_14_1d_aligned[i] >= atr_20th) and (atr_14_1d_aligned[i] <= atr_80th)
        else:
            vol_filter = True  # Not enough data yet
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma_20[i] * 0.7
        
        # Price proximity to pivot levels (within 0.5% of S1 or R1)
        # Long when near S1 support, short when near R1 resistance
        near_support = abs(close[i] - s1_aligned[i]) / close[i] < 0.005
        near_resistance = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
        
        # Entry conditions
        long_condition = near_support and vol_filter and vol_filter
        short_condition = near_resistance and vol_filter and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit when price moves away from pivot level or volatility drops
        elif position == 1 and not near_support:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not near_resistance:
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

name = "12h_PivotPointReversal_VolumeVolatilityFilter"
timeframe = "12h"
leverage = 1.0