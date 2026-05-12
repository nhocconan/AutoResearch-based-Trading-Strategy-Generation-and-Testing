#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1S1_Breakout_12hTrend_Volume
# Hypothesis: On 4h timeframe, use daily Camarilla pivot levels (R1/S1) for entries with 12h EMA50 trend filter and volume confirmation.
# Enter long when price closes above R1 with volume > 1.5x 20-bar average and 12h EMA50 uptrend.
# Enter short when price closes below S1 with volume > 1.5x 20-bar average and 12h EMA50 downtrend.
# Exit when price returns to the daily pivot point (mean reversion) or reverses at opposite level (R1 for longs, S1 for shorts).
# Targets 20-40 trades/year to minimize fee drag while capturing meaningful moves in both bull and bear markets.

name = "4h_Camarilla_Pivot_R1S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Load daily data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R1 and S1 levels (most significant for breakouts)
    r1 = daily_pivot + daily_range * 1.083
    s1 = daily_pivot - daily_range * 1.083
    
    # Align daily levels to 4h timeframe (wait for completed daily bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        ema12h_trend = ema50_12h_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume confirmation and 12h uptrend
            if close[i] > r1_val and close[i] > ema12h_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume confirmation and 12h downtrend
            elif close[i] < s1_val and close[i] < ema12h_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion) or breaks below S1 (invalidates bullish bias)
            if close[i] <= pivot_val or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion) or breaks above R1 (invalidates bearish bias)
            if close[i] >= pivot_val or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals