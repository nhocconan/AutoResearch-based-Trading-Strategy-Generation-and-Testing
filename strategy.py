#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d volume spike and 4h choppiness regime filter.
- Primary timeframe: 4h for execution.
- HTF: 1d for volume confirmation (volume > 1.5 * 20-period volume MA).
- Regime filter: 4h choppiness index (CHOP) > 61.8 = ranging (mean reversion at pivot), CHOP < 38.2 = trending (breakout).
- Entry: In trending (CHOP < 38.2): Long when price breaks above R1 with volume spike.
                         Short when price breaks below S1 with volume spike.
         In ranging (CHOP > 61.8): Long when price touches S1 and reverses up (close > low).
                                  Short when price touches R1 and reverses down (close < high).
- Exit: Opposite pivot breakout or regime shift.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 4h choppiness index (CHOP) - 14 period
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(low.shift())).abs()
    tr3 = (pd.Series(low) - pd.Series(close.shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # For 4h, we use daily OHLC from 1d data
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Typical price
    tp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla width
    width = (prev_high - prev_low) * 1.1 / 12.0
    # R1, S1 levels
    r1 = tp + width * 1.1
    s1 = tp - width * 1.1
    
    # Align 1d Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # Need enough bars for volume MA, CHOP, and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        chop_val = chop[i]
        volume_spike = volume_spike_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike:
                if chop_val < 38.2:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above R1
                    if curr_close > r1_val:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below S1
                    elif curr_close < s1_val:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at extremes
                    # Long when price touches S1 and shows reversal (close > low)
                    if curr_low <= s1_val and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches R1 and shows reversal (close < high)
                    elif curr_high >= r1_val and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below S1 OR regime shifts to trending (CHOP < 38.2)
            if curr_close < s1_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 OR regime shifts to trending (CHOP < 38.2)
            if curr_close > r1_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_4hCHOPRegime_v1"
timeframe = "4h"
leverage = 1.0