#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) Breakout + 1D Trend + Volume Spike
# Hypothesis: Donchian breakouts capture momentum. We filter by 1-day trend (EMA50)
# to avoid counter-trend trades and require volume spike (>2x 20-period average) 
# to confirm breakout strength. This works in both bull (breakouts continue) 
# and bear (breakdowns continue) markets by trading in direction of higher timeframe trend.
# Target: 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag on 6h.
name = "6h_donchian20_1d_trend_volume_spike_v1"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume spike filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or trend turns bearish
            if close[i] < low_20[i] or close[i] < daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or trend turns bullish
            if close[i] > high_20[i] or close[i] > daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume spike and Donchian breakout
            if vol_spike[i]:
                # Long: price breaks above upper band + above 1D EMA (bullish trend)
                if close[i] > high_20[i] and close[i] > daily_ema_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band + below 1D EMA (bearish trend)
                elif close[i] < low_20[i] and close[i] < daily_ema_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals