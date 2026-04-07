#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Daily Trend Filter and Volume Spike
# Hypothesis: Donchian(20) breakouts capture trend continuation. Daily trend filter
# ensures trades align with higher timeframe bias. Volume spike confirms breakout
# strength. Works in bull/bear as breakouts occur in all regimes. Targets 20-40 trades/year.

name = "4h_donchian_breakout_daily_trend_volume_v1"
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
    
    # Donchian Channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter (using 1h data as proxy for daily trend)
    # We'll use 4-period EMA on 4h data (~1 day) for trend direction
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=4, adjust=False).mean().values
    ema_slow = close_series.ewm(span=12, adjust=False).mean().values
    daily_trend_up = ema_fast > ema_slow
    daily_trend_down = ema_fast < ema_slow
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR trend reverses
            if close[i] < donchian_low[i] or not daily_trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR trend reverses
            if close[i] > donchian_high[i] or not daily_trend_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade on Donchian breakout with volume spike and trend alignment
            if vol_spike[i]:
                # Breakout up: close above Donchian high with prior bar below
                if close[i] > donchian_high[i] and (i == 20 or close[i-1] <= donchian_high[i-1]):
                    if daily_trend_up[i]:  # Only long in uptrend
                        position = 1
                        signals[i] = 0.25
                # Breakout down: close below Donchian low with prior bar above
                elif close[i] < donchian_low[i] and (i == 20 or close[i-1] >= donchian_low[i-1]):
                    if daily_trend_down[i]:  # Only short in downtrend
                        position = -1
                        signals[i] = -0.25
    
    return signals