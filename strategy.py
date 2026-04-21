#!/usr/bin/env python3
"""
6h_HTF_WeeklyDonchian_DailyBreakout_V1
Hypothesis: Use weekly Donchian(20) for long-term trend direction + daily breakout of 4h session high/low for entry.
Only trade in direction of weekly trend: long when price > weekly upper band, short when price < weekly lower band.
Enter on 4h breakout of session high (long) or session low (short) with volume confirmation (>1.5x 20-bar MA).
Exit on opposite session breakout or when price crosses weekly midpoint (mean reversion within weekly channel).
Uses discrete sizing (0.25) to limit drawdown in bear markets. Target 12-25 trades/year per symbol.
Works in bull (catch trends) and bear (fade reversals at weekly extremes) via confluence of weekly structure and 4h momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly Donchian
    df_1d = get_htf_data(prices, '1d')  # for daily session high/low
    
    if len(df_1w) < 20 or len(df_1d) < 1:
        return np.zeros(n)
    
    # === Weekly Donchian Channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    weekly_mid = (weekly_upper + weekly_lower) / 2.0
    
    # Align weekly levels to 6h timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # === Daily Session High/Low (from 1d data) ===
    # Use prior day's high/low as reference for 4h breakout
    session_high = df_1d['high'].values
    session_low = df_1d['low'].values
    session_high_aligned = align_htf_to_ltf(prices, df_1d, session_high)
    session_low_aligned = align_htf_to_ltf(prices, df_1d, session_low)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or
            np.isnan(weekly_mid_aligned[i]) or np.isnan(session_high_aligned[i]) or
            np.isnan(session_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price > weekly upper AND 4h breaks above prior day's high with volume
            if price > weekly_upper_aligned[i] and high[i] > session_high_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly lower AND 4h breaks below prior day's low with volume
            elif price < weekly_lower_aligned[i] and low[i] < session_low_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below prior day's low OR crosses weekly midpoint (mean reversion)
            if low[i] < session_low_aligned[i] or price < weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above prior day's high OR crosses weekly midpoint (mean reversion)
            if high[i] > session_high_aligned[i] or price > weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_WeeklyDonchian_DailyBreakout_V1"
timeframe = "6h"
leverage = 1.0