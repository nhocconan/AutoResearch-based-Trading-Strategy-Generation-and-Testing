#!/usr/bin/env python3
# 4h_russell_bull_bear_power_v1
# Hypothesis: Russell's Bull/Bear Power oscillator with EMA13 and volume confirmation on 4h timeframe
# captures institutional buying/selling pressure. Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and rising with volume > 1.5x 20-period average.
# Short when Bear Power > 0 and rising with volume > 1.5x 20-period average.
# Weekly EMA50 trend filter ensures alignment with higher timeframe trend.
# Target: 20-30 trades/year (80-120 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_russell_bull_bear_power_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Smooth the power indicators
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative or weekly trend turns bearish
            if bull_power_smooth[i] <= 0 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns negative or weekly trend turns bullish
            if bear_power_smooth[i] <= 0 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power > 0 and rising with volume confirmation and weekly uptrend
            if bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1] and \
               volume[i] > vol_threshold[i] and close[i] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power > 0 and rising with volume confirmation and weekly downtrend
            elif bear_power_smooth[i] > 0 and bear_power_smooth[i] > bear_power_smooth[i-1] and \
                 volume[i] > vol_threshold[i] and close[i] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals