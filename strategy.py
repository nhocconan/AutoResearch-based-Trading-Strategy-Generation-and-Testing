#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h Trend + Volume Filter
# Hypothesis: Donchian channel breakouts capture strong momentum, filtered by
# 12h trend direction and volume confirmation. Works in both bull and bear
# markets by using trend filter to avoid counter-trend entries.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.
name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Donchian channel (20-period)
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend turns bearish
            if close[i] < low_roll[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend turns bullish
            if close[i] > high_roll[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price breaks above Donchian upper band with bullish trend
                if close[i] > high_roll[i] and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian lower band with bearish trend
                elif close[i] < low_roll[i] and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals