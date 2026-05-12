#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1DVOLUME_CONFIRMATION
# Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d volume confirmation to filter false breakouts.
# Long when 4h close breaks above R1 and 1d volume > 1.5x 20-period average; short when breaks below S1 with same volume filter.
# Uses 1d EMA50 trend filter to avoid counter-trend trades. Designed for low-frequency, high-probability setups in all market regimes.
# Targets 25-40 trades/year to minimize fee drag while capturing institutional interest levels.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1DVOLUME_CONFIRMATION"
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
    
    # Calculate Camarilla levels for each 4h bar using prior bar's OHLC
    # R1 = C + (H-L)*1.12/12, S1 = C - (H-L)*1.12/12
    # We use the previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # Initialize first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.12 / 12
    s1 = prev_close - rang * 1.12 / 12
    
    # 1d volume confirmation: compare current 1d volume to its 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough for 20-period average
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period average
        vol_filter = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        if position == 0:
            # LONG: close breaks above R1 with volume confirmation and uptrend
            if close[i] > r1[i] and vol_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: close breaks below S1 with volume confirmation and downtrend
            elif close[i] < s1[i] and vol_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close returns below R1 or trend breaks
            if close[i] < r1[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close returns above S1 or trend breaks
            if close[i] > s1[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals