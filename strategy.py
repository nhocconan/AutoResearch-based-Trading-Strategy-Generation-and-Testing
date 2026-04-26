#!/usr/bin/env python3
"""
6h_IBS_Pullback_4hTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when IBS (Inner Bar Strength) < 0.2 (oversold pullback) in 4h uptrend (close > EMA34) with volume > 1.5x average. Enter short when IBS > 0.8 (overbought bounce) in 4h downtrend (close < EMA34) with volume spike. Uses discrete position size 0.25. Designed for 15-25 trades/year on 6h by requiring pullback in trend context with volume confirmation, capturing mean reversion within trends while avoiding chop. Works in both bull (buy pullbacks) and bear (sell bounces) markets.
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
    
    # Get 4h data for IBS and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h IBS: (close - low) / (high - low)
    ibs = (df_4h['close'] - df_4h['low']) / np.maximum(df_4h['high'] - df_4h['low'], 1e-10)
    ibs_values = ibs.values
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h indicators to 6h timeframe
    ibs_aligned = align_htf_to_ltf(prices, df_4h, ibs_values)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ibs_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_34_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: IBS < 0.2 (oversold pullback) + 4h uptrend + volume filter
            long_signal = (ibs_aligned[i] < 0.2) and trend_4h_uptrend and volume_filter[i]
            
            # Short: IBS > 0.8 (overbought bounce) + 4h downtrend + volume filter
            short_signal = (ibs_aligned[i] > 0.8) and trend_4h_downtrend and volume_filter[i]
            
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
            # Exit: IBS > 0.8 (overbought) OR 4h trend turns down
            if (ibs_aligned[i] > 0.8 or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: IBS < 0.2 (oversold) OR 4h trend turns up
            if (ibs_aligned[i] < 0.2 or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_IBS_Pullback_4hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0