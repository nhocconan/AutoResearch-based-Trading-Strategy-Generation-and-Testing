#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Touch_Volume_1dTrend_v1
Hypothesis: Uses Camarilla pivot levels (R1/S1) from 1d timeframe with volume confirmation and 1d EMA trend filter.
Takes long when price touches S1 with volume spike in uptrend, short when touches R1 with volume spike in downtrend.
Works in bull markets via trend-following bounces at pivot levels and bear markets via rejection at pivot levels.
Target: 20-40 trades/year to minimize fee drag while capturing mean reversion at institutional levels.
"""

name = "4h_Camarilla_Pivot_Touch_Volume_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (based on previous day) ---
    # Calculate pivot points using previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid first value where shift creates NaN
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe (they are fixed for the day)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d EMA for trend filter ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on EMA
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Touch conditions with tolerance (0.1% to avoid whipsaw)
        touch_s1 = abs(low[i] - s1_4h[i]) / s1_4h[i] < 0.001
        touch_r1 = abs(high[i] - r1_4h[i]) / r1_4h[i] < 0.001
        
        if position == 0:
            # Look for touch with volume confirmation
            if price_above_ema and touch_s1 and vol_spike[i]:
                # Uptrend + touch S1 + volume = long
                signals[i] = 0.25
                position = 1
            elif price_below_ema and touch_r1 and vol_spike[i]:
                # Downtrend + touch R1 + volume = short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches R1 or closes below EMA
                exit_signal = touch_r1 or (close[i] < ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches S1 or closes above EMA
                exit_signal = touch_s1 or (close[i] > ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals