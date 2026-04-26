#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (2.0x).
- Primary timeframe 4h targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above R1 (Camarilla resistance 1) AND 1d uptrend AND volume spike
- Short when price breaks below S1 (Camarilla support 1) AND 1d downtrend AND volume spike
- Uses prior day's OHLC to calculate Camarilla levels (no look-ahead)
- Volume confirmation filters low-participation breakouts
- 1d EMA34 trend filter reduces whipsaw in bear markets and captures major moves
- Discrete position sizing: 0.25 (25% of capital) to balance return and drawdown
- Designed for BTC/ETH with proven edge from Camarilla's intraday pivot effectiveness
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC (no look-ahead)
    # Use shift(1) to ensure we only use completed 1d bars
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_range = prior_high - prior_low
    r1 = prior_close + camarilla_range * 1.0 / 12
    s1 = prior_close - camarilla_range * 1.0 / 12
    
    # Align to 4h timeframe (available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, and 1 for prior day shift)
    start_idx = max(34, 20) + 1
    
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
        
        # Camarilla R1/S1 breakout conditions with volume confirmation and trend filter
        if position == 0:
            # Long: Break above R1 AND 1d uptrend AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 AND 1d downtrend AND volume spike
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

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend_v1"
timeframe = "4h"
leverage = 1.0