#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: Breakouts above Camarilla R1 or below S1 with volume confirmation and 12h EMA34 trend filter.
Camarilla levels provide institutional support/resistance. EMA34 trend filter avoids counter-trend trades.
Volume spike confirms institutional participation. Target: 25-35 trades/year to minimize fee drag.
Works in bull/bear via trend filter and tight entry conditions.
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
    
    # 12h EMA34 for trend filter (loaded once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Camarilla levels from previous day (requires daily data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rango = high_1d - low_1d
    camarilla_r1 = close_1d + rango * 1.1 / 12
    camarilla_s1 = close_1d - rango * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_12h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend (price > 12h EMA34)
            if (price > r1 and vol_spike and price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend (price < 12h EMA34)
            elif (price < s1 and vol_spike and price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below 12h EMA34 OR breaks below S1 (reversal)
            if price < ema34 or price < s1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above 12h EMA34 OR breaks above R1 (reversal)
            if price > ema34 or price > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0