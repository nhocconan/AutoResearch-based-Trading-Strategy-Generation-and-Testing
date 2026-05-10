#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_12h
Hypothesis: Combines 4h Donchian channel breakouts with 12h trend filter and volume confirmation.
Works in bull markets by capturing breakouts above upper band and in bear markets by capturing
breakdowns below lower band. Uses 12h EMA for trend direction and volume spike for confirmation.
Target: 20-40 trades/year per symbol.
"""

name = "4h_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian Channel (20-period)
    donchian_high = high_s.rolling(window=20, min_periods=20).max()
    donchian_low = low_s.rolling(window=20, min_periods=20).min()
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Conditions for long entry
        donchian_breakout = close[i] > donchian_high[i-1]
        
        # Conditions for short entry
        donchian_breakdown = close[i] < donchian_low[i-1]
        
        if position == 0:
            # Enter long: breakout above upper band + 12h uptrend + volume
            if donchian_breakout and trend_12h_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below lower band + 12h downtrend + volume
            elif donchian_breakdown and trend_12h_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price breaks below lower band or trend changes
            if close[i] < donchian_low[i] or trend_12h_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above upper band or trend changes
            if close[i] > donchian_high[i] or trend_12h_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals