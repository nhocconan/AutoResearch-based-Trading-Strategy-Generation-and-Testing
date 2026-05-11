#!/usr/bin/env python3
"""
12h_KAMA_Trend_Volume_Signal_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
In ranging markets, KAMA stays flat, reducing false signals. Combined with
volume confirmation (2x average) and 1d ADX trend filter (>25), this strategy
captures strong trends while avoiding whipsaws. Targets 12-37 trades/year
on 12h timeframe with low frequency to minimize fee drag. Works in both bull
and bear markets by only trading when trend is strong (ADX>25).
"""

name = "12h_KAMA_Trend_Volume_Signal_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (10-period ER, 2/30 SC) on 12h ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, k=1, prepend=0)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- 1d ADX (14-period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first day
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(val, period):
        s = np.zeros_like(val)
        s[period-1] = np.sum(val[:period])
        for i in range(period, len(val)):
            s[i] = s[i-1] - (s[i-1] / period) + val[i]
        return s
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth(dx, 14)
    
    # Align 1d ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if ADX is NaN
        if np.isnan(adx_12h[i]):
            if position != 0:
                # Simple trailing: exit if price crosses KAMA
                if position == 1 and close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: KAMA trend + volume spike + strong trend (ADX>25)
        long_entry = (close[i] > kama[i]) and vol_spike[i] and (adx_12h[i] > 25)
        short_entry = (close[i] < kama[i]) and vol_spike[i] and (adx_12h[i] > 25)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on KAMA cross or ADX weakening
            if position == 1:
                if (close[i] < kama[i]) or (adx_12h[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > kama[i]) or (adx_12h[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals