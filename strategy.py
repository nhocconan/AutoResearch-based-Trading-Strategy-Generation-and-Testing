# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_1w_RSI_Channel_With_Volume_Filter
Hypothesis: On 12h timeframe, when price breaks above/below daily Keltner channels with RSI(14) > 60/<40 and daily volume > 1.5x 20-period average, enter position. Weekly trend filter (price vs weekly SMA50) ensures alignment with higher timeframe trend. Exit on opposite channel touch or trend reversal. Designed for 12h to target 12-37 trades/year with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily Keltner Channel (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate ATR(10) for Keltner Channel
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel: 2 * ATR
    keltner_mid = prev_close
    keltner_upper = keltner_mid + 2 * atr
    keltner_lower = keltner_mid - 2 * atr
    
    # Align 1d Keltner levels to 12h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    keltner_mid_aligned = align_htf_to_ltf(prices, df_1d, keltner_mid)
    
    # Daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Daily volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(keltner_mid_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Weekly trend condition
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > keltner_upper_aligned[i]
        short_breakout = close[i] < keltner_lower_aligned[i]
        
        # RSI conditions
        rsi_overbought = rsi_aligned[i] > 60
        rsi_oversold = rsi_aligned[i] < 40
        
        # Exit conditions
        long_exit = close[i] < keltner_mid_aligned[i]
        short_exit = close[i] > keltner_mid_aligned[i]
        trend_reverse_long = close[i] < sma_50_aligned[i]  # uptrend broken
        trend_reverse_short = close[i] > sma_50_aligned[i]  # downtrend broken
        
        if position == 0:
            if long_breakout and vol_condition and rsi_overbought and uptrend:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and rsi_oversold and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or trend_reverse_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or trend_reverse_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_RSI_Channel_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0