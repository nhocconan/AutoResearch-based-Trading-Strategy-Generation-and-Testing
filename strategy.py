#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Volume
# Hypothesis: Use 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Bull Power (high - EMA) > 0 and Bear Power (low - EMA) < 0 indicate bull/bear pressure.
# Enter long when Bull Power > 0, trend up (price > 1d EMA50), and volume > average.
# Enter short when Bear Power < 0, trend down, and volume > average.
# This captures momentum in trending markets while avoiding chop via volume filter.
# Designed for low frequency (15-30 trades/year) to survive both bull and bear markets.

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA13 for Elder Ray calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Elder Ray components: Bull Power (high - EMA13), Bear Power (low - EMA13) ===
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # === Volume confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0  # Buying pressure
        bear_signal = bear_power[i] < 0  # Selling pressure
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: bullish pressure, uptrend, volume confirmation
            if bull_signal and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish pressure, downtrend, volume confirmation
            elif bear_signal and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: bearish pressure or trend reversal
            if bear_signal or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish pressure or trend reversal
            if bull_signal or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals