#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12H_TREND_VOLUME
# Hypothesis: Camarilla pivot levels (R1/S1) act as strong intraday support/resistance.
# Breakouts above R1 or below S1 with 12h EMA50 trend filter and volume confirmation capture
# momentum moves in both bull and bear markets. Volume filter avoids false breakouts.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12H_TREND_VOLUME"
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
    
    # 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla formulas
    R1 = pc + (ph - pl) * 1.1 / 12
    S1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (with 1-bar delay for completed day)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h data for trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma_12h = pd.Series(df_12h['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and uptrend
            if (close[i] > R1_aligned[i] and vol_confirm and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and downtrend
            elif (close[i] < S1_aligned[i] and vol_confirm and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend reversal
            if close[i] < R1_aligned[i] or close[i] <= ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend reversal
            if close[i] > S1_aligned[i] or close[i] >= ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals