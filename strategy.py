#!/usr/bin/env python3
# 1D_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Uses weekly trend filter (1w EMA12) with daily Camarilla R1/S1 breakouts and volume confirmation.
# Weekly trend ensures alignment with higher timeframe direction, reducing false signals in both bull and bear markets.
# Daily timeframe allows for precise entry/exit while maintaining low trade frequency (target: 10-25 trades/year).
# Volume spike confirms institutional participation. Designed for BTC/ETH with controlled risk via position sizing.

name = "1D_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    # Calculate weekly EMA(12) for trend direction
    ema_12_1w = pd.Series(df_1w['close']).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_12_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + ((H-L) * 1.1 / 12)
    # S1 = C - ((H-L) * 1.1 / 12)
    camarilla_r1 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    camarilla_s1 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume filter: volume > 1.8x 20-period average on daily chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(12, 20)  # Warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_12_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA12
        price_above_weekly_ema = close[i] > ema_12_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_12_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above weekly EMA12 + volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_weekly_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + below weekly EMA12 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_weekly_ema and 
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