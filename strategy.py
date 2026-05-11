#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Points_With_WeeklyTrend
Hypothesis: Camarilla pivot levels on 1d provide strong intraday support/resistance. 
Combined with 1w EMA trend filter to align with higher timeframe momentum. 
Only trade when price touches S3 or R3 levels with volume confirmation and in-trend conditions.
Designed for low turnover (~15-25 trades/year) to minimize fee dust and survive 2025 bear market.
"""

name = "1d_Camarilla_Pivot_Points_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # === Camarilla Pivot Levels (using previous day's OHLC) ===
    # Calculate pivots with 1-day lag to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first day uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    
    # === Weekly Trend Filter (EMA 20 on 1w) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Volume Confirmation (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 20  # covers EMA20 and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(S3[i]) or np.isnan(R3[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Price touch conditions with small buffer
        buffer = 0.001 * close[i]  # 0.1% buffer
        touch_S3 = low[i] <= (S3[i] + buffer) and low[i] >= (S3[i] - buffer)
        touch_R3 = high[i] >= (R3[i] - buffer) and high[i] <= (R3[i] + buffer)
        
        # Trend conditions
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: Price touches S3 level + in uptrend + volume confirmation
            if touch_S3 and uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price touches R3 level + in downtrend + volume confirmation
            elif touch_R3 and downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: price moves to opposite S1/R1 level or reverses
            if position == 1:
                # Exit long: price touches S1 or crosses below pivot
                touch_S1 = low[i] <= (S1[i] + buffer) and low[i] >= (S1[i] - buffer)
                if touch_S1 or close[i] < pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: price touches R1 or crosses above pivot
                touch_R1 = high[i] >= (R1[i] - buffer) and high[i] <= (R1[i] + buffer)
                if touch_R1 or close[i] > pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals