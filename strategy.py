#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above 4h Donchian(20) upper band and price > 1d EMA(50) and volume > 20-bar average.
# Short when price breaks below 4h Donchian(20) lower band and price < 1d EMA(50) and volume > 20-bar average.
# Exit when price crosses the opposite Donchian band or trend reverses.
# Uses tight breakout logic to limit trades to 75-200 total over 4 years.
# Works in bull (breakouts with trend) and bear (breakouts against trend but filtered by EMA).

name = "4h_donchian20_1d_ema_vol_v1"
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
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_1d = df_1d['close'].ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian lower band or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian upper band or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filter
            if vol_filter:
                # Long breakout: price above upper band and above 1d EMA
                if close[i] > high_20[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below lower band and below 1d EMA
                elif close[i] < low_20[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals