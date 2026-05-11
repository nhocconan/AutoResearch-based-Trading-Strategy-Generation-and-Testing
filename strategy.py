# 6th
#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: 6h Donchian breakout in direction of weekly pivot trend, filtered by 1d volume spike.
# Long when price breaks above Donchian high(20) AND weekly pivot trend is up AND volume spike.
# Short when price breaks below Donchian low(20) AND weekly pivot trend is down AND volume spike.
# Exit when price returns to Donchian midpoint or weekly pivot trend reverses.
# Works in bull markets by catching upward breakouts and in bear by catching downward breakdowns.
# Weekly pivot provides directional bias, Donchian gives clear entry/exit, volume confirms strength.

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly pivot calculation (based on previous week) ---
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    resistance_1 = 2 * pivot_point - weekly_low
    support_1 = 2 * pivot_point - weekly_high
    resistance_2 = pivot_point + (weekly_high - weekly_low)
    support_2 = pivot_point - (weekly_high - weekly_low)
    resistance_3 = weekly_high + 2 * (pivot_point - weekly_low)
    support_3 = weekly_low - 2 * (weekly_high - pivot_point)
    
    # Weekly trend: up if close > resistance_1, down if close < support_1
    weekly_trend_up = weekly_close > resistance_1
    weekly_trend_down = weekly_close < support_1
    
    # Align weekly trend to 6h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_down)
    
    # --- Donchian(20) channels ---
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # --- 1d volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and volume MA(20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or
            np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_up and weekly_trend_up_aligned[i] and vol_spike:
                # Long: upward breakout + up weekly trend + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and weekly_trend_down_aligned[i] and vol_spike:
                # Short: downward breakout + down weekly trend + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR weekly trend turns down
                if close[i] < donchian_mid[i] or weekly_trend_down_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR weekly trend turns up
                if close[i] > donchian_mid[i] or weekly_trend_up_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals