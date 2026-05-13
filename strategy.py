#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Weekly Camarilla pivot levels (R1/S1) act as strong support/resistance.
# Buy when price breaks above R1 with volume confirmation and weekly trend up.
# Sell when price breaks below S1 with volume confirmation and weekly trend down.
# Weekly trend filter reduces whipsaws in ranging markets.
# Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend).
# Target: 15-25 trades/year to minimize fee drag.

name = "1d_Weekly_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Calculate weekly Camarilla levels (based on prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Need at least 2 weeks of data to calculate levels for current week
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use prior week's OHLC to calculate current week's levels
    # Weekly high, low, close from previous completed week
    wk_high = df_1w['high'].values[:-1]  # Exclude current week
    wk_low = df_1w['low'].values[:-1]
    wk_close = df_1w['close'].values[:-1]
    
    # Camarilla calculation: 
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    wk_range = wk_high - wk_low
    r1 = wk_close + wk_range * 1.1 / 12
    s1 = wk_close - wk_range * 1.1 / 12
    
    # Align weekly levels to daily timeframe (available after weekly close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter: EMA50 on weekly close
    wk_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align weekly EMA to daily (available after weekly close)
    wk_ema50_aligned = align_htf_to_ltf(prices, df_1w, wk_ema50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 days for vol MA + 50 for EMA)
    start_idx = max(20, 50)
    for i in range(start_idx, n):
        if position == 0:
            # LONG: Break above R1 with volume and weekly uptrend
            if (r1_aligned[i] > 0 and  # Valid level
                close[i] > r1_aligned[i] and
                volume_confirm[i] and
                wk_ema50_aligned[i] > 0 and  # Valid EMA
                close[i] > wk_ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and weekly downtrend
            elif (s1_aligned[i] > 0 and  # Valid level
                  close[i] < s1_aligned[i] and
                  volume_confirm[i] and
                  wk_ema50_aligned[i] > 0 and  # Valid EMA
                  close[i] < wk_ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend weakens
            if close[i] < r1_aligned[i] or close[i] < wk_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend weakens
            if close[i] > s1_aligned[i] or close[i] > wk_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals