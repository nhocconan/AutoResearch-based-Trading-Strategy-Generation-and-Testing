#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE
# Hypothesis: Camarilla R3/S3 breakout with 1-day trend filter and volume spike confirmation.
# Long when price breaks above R3 with volume > 1.5x 20-period average and price above 1-day EMA34.
# Short when price breaks below S3 with volume > 1.5x 20-period average and price below 1-day EMA34.
# Exit when price returns to opposite Camarilla level (S3 for long, R3 for short) or trend reverses.
# Designed for 12-hour timeframe to capture institutional levels with minimal trades (target 15-30/year).
# Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE"
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
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla R3 and S3 levels
    r3 = pc + (ph - pl) * 1.1 / 2
    s3 = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (they update only when new 1d bar forms)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-day EMA34 for trend filter
    ema1d = pd.Series(pc).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume spike and uptrend
            if close[i] > r3_aligned[i] and volume_spike[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume spike and downtrend
            elif close[i] < s3_aligned[i] and volume_spike[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to S3 or trend breaks
            if close[i] < s3_aligned[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to R3 or trend breaks
            if close[i] > r3_aligned[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals