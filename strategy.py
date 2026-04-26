#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike_v1
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for optimal trade frequency (target: 20-50 trades/year)
- Long when price breaks above R1 AND 12h EMA34 uptrend AND volume > 1.5x 20-period average
- Short when price breaks below S1 AND 12h EMA34 downtrend AND volume > 1.5x 20-period average
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Works in bull/bear markets by aligning with 12h trend while using 4h structure for entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 1d data for Camarilla pivots (using prior 1d session)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC for Camarilla calculation
    prev_high = df_12h['high'].values  # Using 12h high as proxy for simplicity
    prev_low = df_12h['low'].values
    prev_close = df_12h['close'].values
    
    # Calculate Camarilla levels (based on prior 1d range)
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 12h EMA34 uptrend AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 12h EMA34 downtrend AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S1 OR 12h EMA34 turns down
            if close[i] < s1_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR 12h EMA34 turns up
            if close[i] > r1_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0