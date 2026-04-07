#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 12h EMA Trend and Volume Spike
# Hypothesis: 4h Donchian(20) breakouts aligned with 12h EMA(30) trend and volume spikes
# capture momentum in both bull and bear markets. 12h trend filter is more responsive than daily
# while still filtering false signals. Volume confirmation ensures breakout strength.
# Target: 25-50 trades/year (100-200 total) to stay within profitable range.

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(30) for trend filter
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # 4h Donchian(20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_30_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above Donchian high in uptrend
                if close[i] > high_20[i] and close[i] > ema_30_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Breakdown below Donchian low in downtrend
                elif close[i] < low_20[i] and close[i] < ema_30_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals