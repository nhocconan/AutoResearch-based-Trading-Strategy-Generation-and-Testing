#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 4h timeframe, use Camarilla pivot R1/S1 breakout with 1d trend filter (close > 1d EMA34) and volume spike (>1.5x 20-period average) for entries. Go long when price breaks above R1 with bullish 1d trend and volume spike. Go short when price breaks below S1 with bearish 1d trend and volume spike. Exit when price re-enters the Camarilla range (between H3 and L3) or 1d trend reverses. Designed for 19-50 trades/year on 4h by requiring multi-timeframe alignment and volume confirmation, reducing fee drag while capturing strong trending moves in both bull and bear markets. Camarilla pivots work well in ranging and trending markets, providing precise entry/exit levels.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d
    # Based on previous day's high, low, close
    prev_high = pd.Series(df_1d['high'].values).shift(1).values
    prev_low = pd.Series(df_1d['low'].values).shift(1).values
    prev_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    # R1 = Close + (High - Low) * 1.1/12
    r1 = prev_close + range_ * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1/12
    s1 = prev_close - range_ * 1.1 / 12
    # H3 = Close + (High - Low) * 1.1/4
    h3 = prev_close + range_ * 1.1 / 4
    # L3 = Close - (High - Low) * 1.1/4
    l3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d data shift + EMA34 warmup + volume MA warmup
    start_idx = max(34, 20) + 1  # +1 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_1d_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_1d_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters Camarilla range (below H3) OR 1d trend turns down
            if (close[i] < h3_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters Camarilla range (above L3) OR 1d trend turns up
            if (close[i] > l3_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0