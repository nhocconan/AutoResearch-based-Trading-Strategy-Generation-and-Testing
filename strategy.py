#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 1.8x 20-period average. Enter short when price breaks below S1 level AND 1d trend is down (close < EMA34) AND volume spike. Uses Camarilla levels from 1d for precise intraday support/resistance, 1d EMA34 for higher timeframe trend alignment, and volume confirmation to filter false breakouts. Designed for low trade frequency (12-25/year) to avoid fee drag while capturing strong institutional moves in both bull and bear markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, 
    # R2 = close + 1.1*(high-low)*1.1/6, R1 = close + 1.1*(high-low)*1.1/12
    # S1 = close - 1.1*(high-low)*1.1/12, S2 = close - 1.1*(high-low)*1.1/6, 
    # S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using previous day's high, low, close
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = prev_high - prev_low
    camarilla_multiplier = 1.1 * range_1d / 12.0  # 1.1/12 factor for R1/S1
    
    R1 = prev_close + camarilla_multiplier
    S1 = prev_close - camarilla_multiplier
    
    # Align Camarilla levels to 6h timeframe (1d -> 6h: 4x multiplier)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 warmup (34), volume MA warmup (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_R1 = close[i] > R1_aligned[i]
        breakout_below_S1 = close[i] < S1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 1d uptrend + volume spike
            long_signal = breakout_above_R1 and trend_uptrend and volume_spike[i]
            
            # Short: price below S1 + 1d downtrend + volume spike
            short_signal = breakout_below_S1 and trend_downtrend and volume_spike[i]
            
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
            if breakout_below_S1 or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend
            if breakout_above_R1 or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0