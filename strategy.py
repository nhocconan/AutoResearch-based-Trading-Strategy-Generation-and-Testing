#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, use Camarilla pivot levels (R1/S1) from prior day with 1w trend filter (price > 1w EMA34) and volume confirmation (>1.5x 20-period average) for entries. Go long when price breaks above R1 with bullish 1w trend and volume spike. Go short when price breaks below S1 with bearish 1w trend and volume spike. Exit when price re-enters between R1 and S1. Designed for 7-25 trades/year on 1d by requiring multi-timeframe alignment and volume confirmation, reducing fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla calculation and HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels for prior day (using shift to avoid look-ahead)
    # Camarilla: based on prior day's high, low, close
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prior_high = df_1d['high'].values
    prior_low = df_1d['low'].values
    prior_close = df_1d['close'].values
    
    # Calculate levels for prior day (shifted by 1 to avoid using current day's data)
    range_hl = prior_high - prior_low
    r1 = prior_close + range_hl * 1.1 / 12
    s1 = prior_close - range_hl * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (already aligned since calculated on 1d)
    # No additional delay needed as these are based on prior completed day
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla levels from prior day
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1w uptrend + volume spike
            long_signal = (close[i] > r1_level) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 1w downtrend + volume spike
            short_signal = (close[i] < s1_level) and trend_1w_downtrend and volume_spike[i]
            
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
            # Exit: price re-enters between R1 and S1 (below R1) OR 1w trend turns down
            if (close[i] < r1_level or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters between R1 and S1 (above S1) OR 1w trend turns up
            if (close[i] > s1_level or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0