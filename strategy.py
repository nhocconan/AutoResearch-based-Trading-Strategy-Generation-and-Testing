#!/usr/bin/env python3
"""
1h_Supertrend_HTF_Filter_VolumeSpike
Hypothesis: On 1h timeframe, use Supertrend(10,3) for entry timing, filtered by 4h/1d trend alignment and volume spike (>2.0x 20-period average). Enter long when Supertrend turns green AND 4h close > EMA50 AND 1d close > EMA50 AND volume spike. Enter short when Supertrend turns red AND 4h close < EMA50 AND 1d close < EMA50 AND volume spike. Uses discrete position size 0.20 to limit fee churn. Designed for 15-37 trades/year on 1h by requiring HTF alignment and volume confirmation, reducing overtrading while capturing trending moves in both bull and bear markets.
"""

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
    
    # Get 4h and 1d data for HTF trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Supertrend calculation on 1h data (period=10, multiplier=3)
    atr_period = 10
    atr_mult = 3
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Final Upper and Lower Bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
            
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = final_lower[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == -1 and final_upper[i] < final_upper[i-1]:
                final_upper[i] = final_upper[i-1]
            if direction[i] == 1 and final_lower[i] > final_lower[i-1]:
                final_lower[i] = final_lower[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # Supertrend signal: 1 when green (uptrend), -1 when red (downtrend)
    supertrend_signal = direction  # 1 = uptrend/green, -1 = downtrend/red
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Supertrend warmup, EMA warmup, volume MA warmup
    start_idx = max(atr_period, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(supertrend_signal[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # HTF trend alignment: both 4h and 1d must agree
        trend_4h_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_50_4h_aligned[i]
        trend_1d_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Supertrend green + 4h uptrend + 1d uptrend + volume spike
            long_signal = (supertrend_signal[i] == 1) and trend_4h_uptrend and trend_1d_uptrend and volume_spike[i]
            
            # Short: Supertrend red + 4h downtrend + 1d downtrend + volume spike
            short_signal = (supertrend_signal[i] == -1) and trend_4h_downtrend and trend_1d_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Supertrend turns red OR HTF trend breaks
            if (supertrend_signal[i] == -1 or not (trend_4h_uptrend and trend_1d_uptrend)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Supertrend turns green OR HTF trend breaks
            if (supertrend_signal[i] == 1 or not (trend_4h_downtrend and trend_1d_downtrend)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Supertrend_HTF_Filter_VolumeSpike"
timeframe = "1h"
leverage = 1.0