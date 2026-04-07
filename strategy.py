#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + 12h Trend Filter + Volume Spike
# Hypothesis: Donchian channel breakouts on 6h timeframe capture strong momentum moves.
# The 12h EMA filter ensures we only trade in the direction of the higher timeframe trend,
# reducing false signals during counter-trend moves. Volume confirmation adds conviction.
# Works in both bull and bear markets by following the 12h trend direction.
# Target: 20-40 trades/year (80-160 total over 4 years).

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channel (20-period) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend turns bearish
            if close[i] < low_min[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend turns bullish
            if close[i] > high_max[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Buy breakout above Donchian high in uptrend
                if close[i] > high_max[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell breakdown below Donchian low in downtrend
                elif close[i] < low_min[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals