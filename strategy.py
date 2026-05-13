#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_Volume_And_Keltner
Hypothesis: KAMA adapts to market noise, filtering false signals in ranging markets while capturing trends.
Combined with Keltner channel breakouts and volume confirmation, this creates a robust trend-following system
that works in both bull and bear markets by avoiding whipsaws. Target: 10-20 trades/year on 1d timeframe.
"""

name = "1d_KAMA_Trend_Filter_With_Volume_And_Keltner"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # Not enough data for first 10 periods
    
    # Volatility: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.nansum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    er[0:10] = 0  # Not enough data
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no alignment needed as we're already on 1d)
    # But we need to ensure we don't use future data, so we'll use shift(1) for signals
    
    # Weekly trend filter: EMA(34) on weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Keltner Channel: 20-period EMA with 2*ATR bands
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Price above Keltner upper, KAMA rising, weekly uptrend, volume confirmation
            if (close[i] > keltner_upper[i] and 
                kama[i] > kama[i-1] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Keltner lower, KAMA falling, weekly downtrend, volume confirmation
            elif (close[i] < keltner_lower[i] and 
                  kama[i] < kama[i-1] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below Keltner middle or KAMA turns down
            if (close[i] < ema20[i]) or (kama[i] < kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above Keltner middle or KAMA turns up
            if (close[i] > ema20[i]) or (kama[i] > kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals