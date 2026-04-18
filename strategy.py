#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance. 
Price breaking above R1 or below S1 with volume confirmation and 12h EMA trend filter captures 
institutional breakout moves. Works in both bull/bear markets as Camarilla adapts to volatility.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), R2 = close + 1.05*(high-low), R1 = close + 1.025*(high-low)
    # S1 = close - 1.025*(high-low), S2 = close - 1.05*(high-low), S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    r1 = close_1d + 1.025 * (high_1d - low_1d)
    s1 = close_1d - 1.025 * (high_1d - low_1d)
    
    # Align to 12h timeframe (wait for daily close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA34 trend filter
    ema_34 = pd.Series(prices['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_filter = prices['volume'].values > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = prices['close'].iloc[i]
        r1_val = r1_12h[i]
        s1_val = s1_12h[i]
        ema_val = ema_34[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1_val and vol_ok and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1_val and vol_ok and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or trend reverses
            if price < s1_val or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or trend reverses
            if price > r1_val or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0