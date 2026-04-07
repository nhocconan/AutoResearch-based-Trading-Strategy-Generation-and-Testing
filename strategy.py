#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) Breakout + 12h Trend Filter + Volume Spike
# Hypothesis: Breakouts in direction of 12h trend with volume confirmation work in both bull and bear markets.
# Uses Donchian channels for breakout detection, 12h EMA for trend filter, and volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_donchian20_12h_trend_volume_v1"
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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian(20) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_20_6h[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches opposite Donchian band or trend changes
            if low[i] <= low_20[i] or close[i] < ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches opposite Donchian band or trend changes
            if high[i] >= high_20[i] or close[i] > ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of 12h trend with volume spike
            if vol_spike[i]:
                if close[i] > ema_20_6h[i]:  # Uptrend
                    if high[i] > high_20[i]:  # Break above upper band
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < low_20[i]:  # Break below lower band
                        position = -1
                        signals[i] = -0.25
    
    return signals