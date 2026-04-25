#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, enter long when price touches Camarilla S1 support with 4h uptrend and volume spike; enter short when price touches R1 resistance with 4h downtrend and volume spike. Use 4h EMA50 as trend filter to avoid counter-trend trades. Target 15-35 trades/year by requiring confluence of 4h trend, 1h Camarilla level touch, and volume spike. Works in bull markets via trend-following breakouts and in bear markets via mean reversion at Camarilla levels within the trend filter.
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
    
    # 4h data for trend filter and Camarilla calculation (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Camarilla pivot levels (based on previous 4h bar's OHLC)
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_range_4h = prev_high_4h - prev_low_4h
    
    H3 = prev_close_4h + 1.125 * prev_range_4h
    L3 = prev_close_4h - 1.125 * prev_range_4h
    H4 = prev_close_4h + 1.5 * prev_range_4h
    L4 = prev_close_4h - 1.5 * prev_range_4h
    R1 = prev_close_4h + 1.071 * prev_range_4h
    S1 = prev_close_4h - 1.071 * prev_range_4h
    
    # Align 4h levels to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    H4_aligned = align_htf_to_ltf(prices, df_4h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_4h, L4)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 4h indicators (50 for EMA, 1 for shift)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_low = low[i]
        curr_high = high[i]
        
        # Trend filter
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price touches S1 support in uptrend with volume spike
            long_entry = (curr_low <= S1_aligned[i]) and uptrend and volume_spike[i]
            # Short: price touches R1 resistance in downtrend with volume spike
            short_entry = (curr_high >= R1_aligned[i]) and downtrend and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on mean reversion or trend change
            # Exit when price reaches midpoint (neutral) or trend fails
            midpoint = (S1_aligned[i] + R1_aligned[i]) / 2
            if curr_close >= midpoint or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on mean reversion or trend change
            midpoint = (S1_aligned[i] + R1_aligned[i]) / 2
            if curr_close <= midpoint or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0