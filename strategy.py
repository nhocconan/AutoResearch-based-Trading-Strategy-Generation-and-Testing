#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for lower trade frequency and reduced fee drag.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian Channel: 20-period high/low for breakout signals.
- Entry: Long when price breaks above Donchian(20) high AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 12h EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout (price < Donchian(20) low for long, price > Donchian(20) high for short).
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This strategy captures strong momentum moves in the direction of the higher timeframe trend,
with volume confirmation to filter false breakouts. Works in both bull and bear markets by
only taking trades aligned with the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 12h volume MA
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need enough bars for Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian high AND 12h EMA50 bullish (close > EMA)
                if curr_high > highest_20[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 12h EMA50 bearish (close < EMA)
                elif curr_low < lowest_20[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if curr_low < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if curr_high > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0