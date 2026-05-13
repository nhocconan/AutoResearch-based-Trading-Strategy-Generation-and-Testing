#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In bull markets, buy when bull power > 0 and rising; in bear markets, sell when bear power < 0 and falling.
# Combined with 1d EMA50 trend filter to avoid counter-trend trades and volume confirmation for breakout strength.
# Designed for low trade frequency (~15-25/year) to minimize fee drift, works in both bull and bear via trend alignment.

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (using close prices)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema13[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull power positive and rising, uptrend, volume confirmation
            if (bull_power[i] > 0 and 
                i > 50 and bull_power[i] > bull_power[i-1] and  # rising bull power
                close[i] > ema50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power negative and falling, downtrend, volume confirmation
            elif (bear_power[i] < 0 and 
                  i > 50 and bear_power[i] < bear_power[i-1] and  # falling bear power (more negative)
                  close[i] < ema50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative (momentum loss) or trend breaks
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns positive (momentum loss) or trend breaks
            if bear_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals