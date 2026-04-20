#!/usr/bin/env python3
# 12h_KAMA_Direction_With_Volume_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend line that reduces whipsaws.
# In choppy markets (high Chop index), KAMA stays flat, avoiding false signals.
# In trending markets (low Chop), KAMA follows price, and breakouts above/below KAMA with volume
# confirmation signal institutional interest. Chop filter ensures we only trade in clear trends.
# Volume confirmation filters false breakouts. Works in both bull and bear via directional breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_KAMA_Direction_With_Volume_and_Chop_Filter"
timeframe = "12h"
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
    
    # Get 1d data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chop index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14)
    atr_values = np.zeros(len(df_1d))
    tr_values = true_range(df_1d['high'].values, df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr_values[0] = df_1d['high'][0] - df_1d['low'][0]  # first TR
    
    # Wilder's smoothing for ATR
    atr_values[0] = tr_values[0]
    for i in range(1, len(tr_values)):
        atr_values[i] = (atr_values[i-1] * 13 + tr_values[i]) / 14
    
    # Calculate Chop: 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_values = np.zeros(len(df_1d))
    for i in range(13, len(df_1d)):
        if i >= 13:
            sum_atr = np.sum(atr_values[i-13:i+1])
            max_high = np.max(df_1d['high'].values[i-13:i+1])
            min_low = np.min(df_1d['low'].values[i-13:i+1])
            if max_high > min_low:
                chop_values[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
            else:
                chop_values[i] = 50  # neutral
    
    # Chop < 38.2 = trending, Chop > 61.8 = ranging
    chop_trending = chop_values < 38.2
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Calculate KAMA (2-period ER, 10-period FC, 30-period SC)
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # SC = [ER * (fastest - slowest) + slowest]^2
    # where fastest = 2/(2+1), slowest = 2/(30+1)
    close_series = pd.Series(close)
    change = np.abs(close_series - close_series.shift(10))
    volatility = np.abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when Chop indicates trending market
        if not chop_aligned[i]:
            # In chop, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume confirmation
            if close[i] > kama[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation
            elif close[i] < kama[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals