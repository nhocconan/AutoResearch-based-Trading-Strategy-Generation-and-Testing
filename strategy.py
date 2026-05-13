#!/usr/bin/env python3
# 12h_KAMA_Trend_Direction_Volume
# Hypothesis: Use KAMA to capture trend direction on 12h timeframe, confirmed by 1d trend and volume spikes.
# KAMA adapts to market noise, reducing whipsaws in choppy conditions while capturing strong trends.
# Long when KAMA is rising and price above KAMA, aligned with 1d uptrend and volume confirmation.
# Short when KAMA is falling and price below KAMA, aligned with 1d downtrend and volume confirmation.
# Works in bull (trend-following) and bear (adaptive filtering reduces false signals).

name = "12h_KAMA_Trend_Direction_Volume"
timeframe = "12h"
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

    # Get daily data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))  # |close - close[er_length]|
    change = np.concatenate([np.full(er_length, np.nan), change])
    
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |diff| over er_length window
    volatility = pd.Series(volatility).rolling(window=er_length, min_periods=er_length).sum().values
    volatility = np.concatenate([np.full(er_length-1, np.nan), volatility[er_length-1:]])
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # seed
    
    for i in range(er_length + 1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: rising if current > previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: volume > 1.5 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.5 * vol_ma_24
    
    # Align daily indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    kama_rising_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama_falling.astype(float))
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(kama_rising_aligned[i]) or 
            np.isnan(kama_falling_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA + KAMA rising + daily uptrend + volume spike
            if close[i] > kama_aligned[i] and kama_rising_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA + KAMA falling + daily downtrend + volume spike
            elif close[i] < kama_aligned[i] and kama_falling_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR trend reversal
            if close[i] < kama_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR trend reversal
            if close[i] > kama_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals