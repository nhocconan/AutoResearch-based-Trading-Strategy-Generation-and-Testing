#!/usr/bin/env python3
"""
6h Camarilla Pivot R1/S1 Breakout with Volume and 1W Trend Filter
Hypothesis: Camarilla pivot levels from daily data provide robust intraday support/resistance.
Breaking above R1 or below S1 with volume confirmation and aligned with weekly trend
(captured via 1w EMA34) captures strong momentum moves while filtering counter-trend noise.
Works in bull/bear by requiring trend alignment. Target: 15-30 trades/year.
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
    
    # === Daily Camarilla Pivots (using prior day's data) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = pc + (ph - pl) * 1.1 / 12
    camarilla_s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 6t timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Weekly Trend Filter (EMA34 on weekly close) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Confirmation (6t) ===
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 34)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        weekly_ema = ema34_1w_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1, above weekly EMA (uptrend), volume confirmation
            if price > r1 and price > weekly_ema and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below weekly EMA (downtrend), volume confirmation
            elif price < s1 and price < weekly_ema and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly EMA or volume weakens
            if price < weekly_ema or vol_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly EMA or volume weakens
            if price > weekly_ema or vol_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_Volume_1WTrend"
timeframe = "6h"
leverage = 1.0