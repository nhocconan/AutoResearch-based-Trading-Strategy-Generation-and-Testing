#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation
# Hypothesis: Donchian breakouts capture strong moves; 1d EMA filters for trend direction
# and volume confirmation reduces false breakouts. Works in bull (breakouts up) and bear
# (breakouts down) markets. Target: 20-50 trades/year (80-200 over 4 years) to minimize
# fee drag. Uses discrete position sizes (0.0, ±0.25) to reduce churn.
name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend turns bearish
            if close[i] < low_20[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend turns bullish
            if close[i] > high_20[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and breakout in trend direction
            if vol_filter[i]:
                # Long: price breaks above upper Donchian + above 1D EMA
                if close[i] > high_20[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian + below 1D EMA
                elif close[i] < low_20[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals