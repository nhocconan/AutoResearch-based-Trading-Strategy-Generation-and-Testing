#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Pullback_Volume_Trend
Hypothesis: Camarilla pivot S1/R1 levels on 1d act as strong support/resistance. 
Buy when price pulls back to S1 in uptrend with volume confirmation. 
Sell when price pulls back to R1 in downtrend with volume confirmation.
Uses 1w EMA40 for trend filter to avoid counter-trend trades. 
Target: 20-30 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA40 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Daily Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_40_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema40 = ema_40_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price pulls back to S1 with volume spike in uptrend
            if (low[i] <= s1 <= high[i] and   # price touches S1
                close[i] > s1 and             # closes above S1 (confirms bounce)
                vol_spike and
                price > ema40):               # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to R1 with volume spike in downtrend
            elif (low[i] <= r1 <= high[i] and   # price touches R1
                  close[i] < r1 and             # closes below R1 (confirms rejection)
                  vol_spike and
                  price < ema40):               # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below S1 or trend reverses
            if close[i] < s1 or price < ema40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above R1 or trend reverses
            if close[i] > r1 or price > ema40:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Pullback_Volume_Trend"
timeframe = "4h"
leverage = 1.0