#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_TrendV1
Hypothesis: Camarilla pivot levels (R1/S1) from prior 1d capture intraday reversals in ranging markets,
while 1d EMA34 filters trend direction to avoid counter-trend trades. Volume spike confirms momentum.
Designed for 15-30 trades/year to minimize fee drag while capturing high-probability reversals.
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
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla pivot levels (R1, S1)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    camarilla_r1 = prev_close + (1.1 * (prev_high - prev_low) / 12)
    camarilla_s1 = prev_close - (1.1 * (prev_high - prev_low) / 12)
    
    # Align to 4h timeframe (previous day's levels available at 00:00 UTC)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price touches S1 in uptrend with volume spike
            if price <= s1 and price > ema34 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 in downtrend with volume spike
            elif price >= r1 and price < ema34 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses EMA34 down OR touches R1 (contrarian exit)
            if price < ema34 or price >= r1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses EMA34 up OR touches S1 (contrarian exit)
            if price > ema34 or price <= s1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_TrendV1"
timeframe = "4h"
leverage = 1.0