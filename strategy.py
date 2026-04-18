#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_DailyTrend
Hypothesis: Buy when price breaks above daily Camarilla R1 with volume spike and above daily EMA50; short when breaks below S1 with volume spike and below daily EMA50. Daily timeframe provides robust trend filter, reducing false breakouts. Volume confirms institutional participation. Designed for low trade frequency (target: 15-30 trades/year) to minimize fee decay while capturing high-probability breakouts in trending and ranging markets.
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
    
    # Daily Camarilla pivot levels from previous day
    df_1d = get_htf_data(prices, '1d')
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rang = phigh - plow
    r1 = pclose + rang * 1.1 / 12
    s1 = pclose - rang * 1.1 / 12
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA50 trend filter
    ema_1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: >2.0x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        ema_1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price > R1 with volume spike and above daily EMA50
            if price > r1_val and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below daily EMA50
            elif price < s1_val and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or below daily EMA50
            if price < s1_val or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or above daily EMA50
            if price > r1_val or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_DailyTrend"
timeframe = "12h"
leverage = 1.0