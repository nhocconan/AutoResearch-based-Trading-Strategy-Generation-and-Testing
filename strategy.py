#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Enters long when price breaks above Camarilla R1 level with 4h bullish trend (close > EMA50) and 1d volume > 1.5x MA20.
# Enters short when price breaks below Camarilla S1 level with 4h bearish trend (close < EMA50) and 1d volume > 1.5x MA20.
# Exits when price reverts to the Camarilla pivot point.
# Uses discrete position sizing (0.20) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~15-37/year) by requiring strict confluence: price breakout + HTF trend + volume confirmation.
# Camarilla R1/S1 levels provide strong intraday support/resistance, while 4h EMA50 filter ensures alignment with higher timeframe momentum.
# Volume confirmation on 1d timeframe reduces false breakouts, improving signal quality in both bull and bear markets.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # Calculate 20-period MA on 1d volume
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values  # previous day high
    prev_low = df_1d['low'].shift(1).values    # previous day low
    prev_close = df_1d['close'].shift(1).values # previous day close
    
    # Calculate Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)  # Resistance 1
    s1 = pivot - (range_hl * 1.1 / 12.0)  # Support 1
    r4 = pivot + (range_hl * 1.1 / 2.0)   # Resistance 4
    s4 = pivot - (range_hl * 1.1 / 2.0)   # Support 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    volume_spike = volume_1d > (vol_ma20_1d * 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with 4h bullish trend and 1d volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 with 4h bearish trend and 1d volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals