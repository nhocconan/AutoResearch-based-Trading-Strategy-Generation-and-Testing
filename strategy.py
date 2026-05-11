#!/usr/bin/env python3
# 12h_1w_1d_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Uses 1w trend filter (price > 1w EMA200) and 1d Camarilla pivot levels (R1/S1) for breakout entries.
# In strong weekly trends, price often pulls back to daily pivot levels before continuing.
# Long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend.
# Volume confirmation ensures breakout strength. Works in bull (buy dips) and bear (sell rallies).
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_1w_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter and 1d data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w EMA200 for trend filter ---
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # --- 1d Camarilla pivot levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_ = high_1d - low_1d
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA200 (200) and volume MA (20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of weekly trend
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 0:
            if uptrend and vol_confirmed:
                # Long: price breaks above R1 in uptrend
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirmed:
                # Short: price breaks below S1 in downtrend
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price closes below pivot OR reverses to S1
                if close[i] < pivot[i] or close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above pivot OR reverses to R1
                if close[i] > pivot[i] or close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals