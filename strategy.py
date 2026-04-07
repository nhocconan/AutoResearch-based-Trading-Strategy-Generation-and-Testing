#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian breakout + 12h volume filter + daily trend filter
# Hypothesis: Donchian(20) breakouts on 4h timeframe capture strong momentum moves.
# Volume confirmation from 12h ensures breakouts are supported by participation.
# Daily EMA(50) filter ensures we only trade in the direction of the higher timeframe trend.
# This combination reduces false breakouts and works in both bull and bear markets.
# Target: 20-40 trades/year to minimize fee drag.
name = "4h_donchian_breakout_12h_volume_daily_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (for volume confirmation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_avg = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume[i]) or np.isnan(volume_12h_avg[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or daily trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or daily trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x 12h average volume
            volume_ok = volume[i] > 1.5 * volume_12h_avg[i]
            
            # Enter long: price breaks above Donchian upper band + volume + bullish daily trend
            if close[i] > highest_high[i] and volume_ok and close[i] > ema_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band + volume + bearish daily trend
            elif close[i] < lowest_low[i] and volume_ok and close[i] < ema_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals