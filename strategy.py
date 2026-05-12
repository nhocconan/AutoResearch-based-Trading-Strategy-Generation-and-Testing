#!/usr/bin/env python3
# 1d_KAMA_Trend_With_WeeklyVWAP_Filter
# Hypothesis: Use daily KAMA direction for trend, weekly VWAP as institutional support/resistance, and volume spike for confirmation.
# Long when KAMA is rising, price > weekly VWAP, and volume > 1.5x 20-day average.
# Short when KAMA is falling, price < weekly VWAP, and volume > 1.5x 20-day average.
# Exit when KAMA direction changes or price crosses weekly VWAP.
# Designed to capture trending moves with institutional level confirmation, works in both bull and bear markets.
# Targets 10-20 trades/year to minimize fee drag on daily timeframe.

name = "1d_KAMA_Trend_With_WeeklyVWAP_Filter"
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.concatenate([[np.nan], volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 for rising, -1 for falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0
    
    # Weekly VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate VWAP for each week: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = typical_price * df_1w['volume']
    cum_pv = pv.cumsum()
    cum_vol = df_1w['volume'].cumsum()
    vwap = cum_pv / cum_vol
    vwap_values = vwap.values
    
    # Align weekly VWAP to daily timeframe (holds value until next weekly update)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # Volume confirmation: 20-day moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising, price above weekly VWAP, volume spike
            if kama_dir[i] == 1 and close[i] > vwap_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, price below weekly VWAP, volume spike
            elif kama_dir[i] == -1 and close[i] < vwap_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or price crosses below VWAP
            if kama_dir[i] == -1 or close[i] < vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or price crosses above VWAP
            if kama_dir[i] == 1 or close[i] > vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals