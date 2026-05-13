#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1w trend filter and volume confirmation
# Enters long when 6h Bull Power > 0, 1w EMA50 trend is up (close > EMA50), and volume > 1.3x MA20
# Enters short when 6h Bear Power < 0, 1w EMA50 trend is down (close < EMA50), and volume > 1.3x MA20
# Exits when Elder Power reverses sign or volume drops below average
# Uses discrete position sizing (0.25) to limit fee churn
# Elder Ray Power measures bull/bear strength relative to EMA13, effective in both trending and ranging markets
# Weekly trend filter ensures alignment with higher timeframe direction, reducing counter-trend trades
# Volume confirmation adds conviction to breakouts
# Designed for low trade frequency (~12-37/year) by requiring confluence: Elder Power signal + HTF trend + volume spike

name = "6h_ElderRay_Power_1wTrend_Volume_v1"
timeframe = "6h"
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Elder Ray Power on 6h timeframe
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bulls in control), 1w uptrend, volume spike
            if bull_power[i] > 0 and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bears in control), 1w downtrend, volume spike
            elif bear_power[i] < 0 and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR volume drops below average
            if bull_power[i] <= 0 or volume[i] <= vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive OR volume drops below average
            if bear_power[i] >= 0 or volume[i] <= vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals