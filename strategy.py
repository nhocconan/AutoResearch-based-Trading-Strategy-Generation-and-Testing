#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Trend_Filter
Hypothesis: Camarilla pivot breakouts on daily timeframe with weekly trend filter and volume confirmation.
In bull markets, buy breaks above R3/R4; in bear markets, sell breaks below S3/S4.
Weekly trend filter ensures we only trade with the higher timeframe direction.
Volume confirmation reduces false breakouts.
Target: 15-25 trades/year per symbol.
"""

name = "1d_1w_Camarilla_Pivot_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily typical price for pivot calculation
    typical_price = (high + low + close) / 3.0
    
    # Calculate Camarilla levels using previous day's data
    # We need to shift by 1 to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first day's values to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Pivot point = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Range = H - L
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R4 = close + (range_val * 1.5000)  # Using current close as base
    R3 = close + (range_val * 1.2500)
    R2 = close + (range_val * 1.1666)
    R1 = close + (range_val * 1.0833)
    S1 = close - (range_val * 1.0833)
    S2 = close - (range_val * 1.1666)
    S3 = close - (range_val * 1.2500)
    S4 = close - (range_val * 1.5000)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    weekly_close = df_1w['close'].values
    ema_20_weekly = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = weekly_close > ema_20_weekly
    downtrend_1w = weekly_close < ema_20_weekly
    
    # Align weekly trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip first day due to roll
        if i == 0:
            continue
            
        if position == 0:
            # LONG: price breaks above R3 with weekly uptrend and volume
            if close[i] > R3[i] and uptrend_1w_aligned[i] and volume_conf[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with weekly downtrend and volume
            elif close[i] < S3[i] and downtrend_1w_aligned[i] and volume_conf[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches S1 or weekly trend turns down
            if close[i] < S1[i] or not uptrend_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches R1 or weekly trend turns up
            if close[i] > R1[i] or not downtrend_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals