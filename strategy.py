#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + 1w Trend Filter + Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts capture strong momentum, filtered by weekly trend (EMA50).
# Volume confirmation ensures institutional participation. Works in bull/bear as Donchian adapts to volatility.
# Target: 20-30 trades/year (80-120 total over 4 years).

name = "1d_donchian_breakout_1w_volume_trend_v1"
timeframe = "1d"
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
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=10).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend turns bearish
            if close[i] < low_roll[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend turns bullish
            if close[i] > high_roll[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout above Donchian upper with volume and bullish trend
            if vol_spike[i] and close[i] > high_roll[i] and (i == 60 or close[i-1] <= high_roll[i-1]) and close[i] > ema50_1d[i]:
                position = 1
                signals[i] = 0.25
            # Breakdown below Donchian lower with volume and bearish trend
            elif vol_spike[i] and close[i] < low_roll[i] and (i == 60 or close[i-1] >= low_roll[i-1]) and close[i] < ema50_1d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals