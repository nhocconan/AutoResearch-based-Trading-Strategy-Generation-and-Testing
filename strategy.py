#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + Weekly Trend + Volume Spike
# Hypothesis: 12h Donchian breakouts aligned with weekly EMA(20) trend and volume spikes
# capture momentum in both bull and bear markets. Weekly trend filter reduces whipsaws
# in sideways markets, volume confirms breakout strength. Target: 15-30 trades/year (60-120 total).

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 12h Donchian(20) - use 20 lookback for 12h period
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above Donchian high in uptrend
                if close[i] > high_20[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Breakdown below Donchian low in downtrend
                elif close[i] < low_20[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals