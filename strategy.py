#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels (R1, S1) act as key support/resistance on 1d chart.
Breakouts above R1 or below S1 with volume confirmation and weekly trend alignment
capture momentum moves. Works in bull markets (buying breakouts) and bear markets
(selling breakdowns) by aligning with weekly trend. Uses 1d for signal generation
and 1w for trend filter. Target: 15-25 trades/year per symbol.
"""

name = "1d_Camarilla_Pivot_Breakout_1wTrend_Volume"
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
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Use previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else high[0]
    prev_low[0] = prev_low[1] if n > 1 else low[0]
    prev_close[0] = prev_close[1] if n > 1 else close[0]
    
    pivot_range = prev_high - prev_low
    r1 = prev_close + pivot_range * 1.1 / 12
    s1 = prev_close - pivot_range * 1.1 / 12
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1w trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R1 with volume and weekly uptrend
            if (close[i] > r1[i] and volume_confirm and trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume and weekly downtrend
            elif (close[i] < s1[i] and volume_confirm and trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns below R1 or weekly trend turns down
            if (close[i] < r1[i] or trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns above S1 or weekly trend turns up
            if (close[i] > s1[i] or trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals