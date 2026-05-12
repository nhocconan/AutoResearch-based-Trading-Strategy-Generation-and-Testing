#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R1, S1) from daily high/low/close act as intraday support/resistance.
# In 1d uptrend (EMA34), go long when price breaks above R1; in downtrend, go short when price breaks below S1.
# Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts.
# Works in both bull and bear markets: trend filter avoids counter-trend trades.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_FILTER"
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
    
    # Daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # where C, H, L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle first day
    rng = prev_high - prev_low
    rng[rng == 0] = 1e-10  # small value to avoid division by zero
    
    r1 = prev_close + rng * 1.1 / 12.0
    s1 = prev_close - rng * 1.1 / 12.0
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R1 + volume confirmation
            if (close[i] > ema34_aligned[i] and 
                close[i] > r1_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S1 + volume confirmation
            elif (close[i] < ema34_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price falls below S1 (invalidates bullish setup)
            if (close[i] <= ema34_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price rises above R1 (invalidates bearish setup)
            if (close[i] >= ema34_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals