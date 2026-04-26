#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, enter long when price breaks above weekly Camarilla R1 level AND weekly trend is up (close > weekly EMA34) AND volume > 2.0x 20-day average. Enter short when price breaks below S1 level AND weekly trend is down (close < weekly EMA34) AND volume spike. Uses weekly Camarilla levels (R1/S1) for structural breakouts with weekly EMA34 trend filter and volume confirmation. Designed for low trade frequency (7-25/year) with edge in both bull and bear markets via trend alignment and volatility-based entry. Uses 1d primary timeframe and 1w HTF as specified in experiment #91644.
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
    
    # Get 1w data for EMA34 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w Camarilla Pivot Levels (R1, S1) from 1w data
    # Based on previous 1w bar's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R1 = close + ((high-low)*1.1/12), S1 = close - ((high-low)*1.1/12)
    camarilla_r1 = close_1w + ((high_1w - low_1w) * 1.1 / 12)
    camarilla_s1 = close_1w - ((high_1w - low_1w) * 1.1 / 12)
    
    # Align Camarilla levels to 1d timeframe (use previous 1w bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), volume MA warmup (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 1w trend filter
        trend_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 1w uptrend + volume spike
            long_signal = breakout_above_r1 and trend_uptrend and volume_spike[i]
            
            # Short: price below S1 + 1w downtrend + volume spike
            short_signal = breakout_below_s1 and trend_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below S1 OR trend change to downtrend
            if close[i] < camarilla_s1_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend
            if close[i] > camarilla_r1_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0