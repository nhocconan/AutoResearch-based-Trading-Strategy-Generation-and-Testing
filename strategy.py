#!/usr/bin/env python3
# 1D_WeeklyPivot_Resistance_Breakout_TrendFilter
# Hypothesis: Buy when price breaks above weekly pivot resistance (R1) with volume confirmation and weekly trend filter (price > weekly EMA50).
# Sell when price breaks below weekly pivot support (S1) or volume drops.
# Uses weekly timeframe for structure and trend, daily for execution to avoid overtrading.
# Designed to capture strong trending moves while avoiding chop, suitable for both bull and bear markets.
# Targets 10-25 trades per year on daily timeframe.

name = "1D_WeeklyPivot_Resistance_Breakout_TrendFilter"
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
    
    # Get weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate weekly pivot points (using prior weekly bar's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # Resistance 1: R1 = 2*P - L
    # Support 1: S1 = 2*P - H
    weekly_high = df_weekly['high']
    weekly_low = df_weekly['low']
    weekly_close = df_weekly['close']
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    resistance_1 = 2 * pivot - weekly_low
    support_1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to daily timeframe (use prior weekly bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, resistance_1.values)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, support_1.values)
    
    # Volume filter: volume > 1.5x 50-period average on daily chart
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_weekly_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_weekly_aligned[i]
        price_below_ema = close[i] < ema_50_weekly_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above weekly EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + below weekly EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S1 (re-enters range) or volume drops below average
            if (close[i] < s1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R1 (re-enters range) or volume drops below average
            if (close[i] > r1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals