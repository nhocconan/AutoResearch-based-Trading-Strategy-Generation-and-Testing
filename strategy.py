#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend
# Hypothesis: Camarilla pivot levels (R3/S3) on daily timeframe act as strong support/resistance.
# Breakout above R3 with 1d uptrend and volume confirmation = long; breakdown below S3 with 1d downtrend and volume confirmation = short.
# Uses 12h timeframe for lower frequency (target 12-37 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) with trend filter.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "12h"
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
    
    # Calculate Camarilla levels for previous day (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = d_close + (d_high - d_low) * 1.1 / 4
    S3 = d_close - (d_high - d_low) * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA for trend filter
    ema1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Volume confirmation: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: Break above R3 with uptrend and volume confirmation
            if close[i] > R3_aligned[i] and close[i] > ema1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with downtrend and volume confirmation
            elif close[i] < S3_aligned[i] and close[i] < ema1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R3 or trend breaks
            if close[i] < R3_aligned[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S3 or trend breaks
            if close[i] > S3_aligned[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals