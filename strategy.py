#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Targets 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Long when price breaks above R1 AND 1d uptrend AND volume spike
- Short when price breaks below S1 AND 1d downtrend AND volume spike
- Camarilla levels provide statistically significant intraday support/resistance
- 1d EMA34 trend filter reduces whipsaw in bear markets and captures major moves
- Volume spike (1.8x 20-period average) confirms institutional participation
- Designed for lower frequency with proven edge on BTC/ETH from Camarilla's mean-reversion-breaking structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (based on previous day's range)
    # For 12h timeframe, we use daily high/low/close from 1d data
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe (already completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike (20-period volume average on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        if position == 0:
            # Long: Price breaks above R1 AND 1d uptrend AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND 1d downtrend AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below S1 OR 1d trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above R1 OR 1d trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_v1"
timeframe = "12h"
leverage = 1.0