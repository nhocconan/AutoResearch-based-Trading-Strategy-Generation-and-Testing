#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume + Trend Filter
# Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and trend alignment
# capture directional moves in both bull and bear markets. The trend filter (1d EMA50)
# avoids counter-trend trades, while volume confirms institutional participation.
# Target: 20-30 trades/year per symbol.

name = "4h_donchian_breakout_volume_trend_v1"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=10).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish
            if close[i] < low_20[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish
            if close[i] > high_20[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume and trend alignment
            if vol_spike[i]:
                # Buy breakout above Donchian high with bullish trend
                if close[i] > high_20[i] and (i == 50 or close[i-1] <= high_20[i-1]) and close[i] > ema50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell breakout below Donchian low with bearish trend
                elif close[i] < low_20[i] and (i == 50 or close[i-1] >= low_20[i-1]) and close[i] < ema50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals