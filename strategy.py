#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_v1
Hypothesis: Use 4h Donchian channel breakout with 1d trend bias and volume confirmation.
Long when 4h price breaks above Donchian(20) high with 1d bullish trend and volume.
Short when 4h price breaks below Donchian(20) low with 1d bearish trend and volume.
Designed to capture breakouts with trend alignment, working in both bull and bear markets.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) by requiring strong breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend bias: close > EMA(50) for bullish, close < EMA(50) for bearish
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = close_1d > ema_50
    trend_bearish = close_1d < ema_50
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or 1d trend turns bearish
            if close[i] < low_roll[i] or trend_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or 1d trend turns bullish
            if close[i] > high_roll[i] or trend_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with 1d bullish trend and volume
            if close[i] > high_roll[i] and trend_bullish_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with 1d bearish trend and volume
            elif close[i] < low_roll[i] and trend_bearish_aligned[i] > 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals