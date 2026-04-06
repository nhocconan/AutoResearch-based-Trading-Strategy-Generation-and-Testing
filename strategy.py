#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Enter long when price breaks above Donchian upper band + price > 1d EMA(50) + volume > 2x average
# Enter short when price breaks below Donchian lower band + price < 1d EMA(50) + volume > 2x average
# Exit on opposite Donchian band touch or trend reversal (price crosses 1d EMA)
# Target: 50-150 trades over 4 years by requiring strong breakouts with volume and trend alignment
# Works in bull markets via breakouts, in bear markets via short breakdowns with trend filter

name = "12h_donchian20_1dema_vol_breakout_v1"
timeframe = "12h"
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
    
    # Donchian channels (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches lower Donchian band OR price crosses below 1d EMA(50)
            if close[i] <= donchian_lower[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian band OR price crosses above 1d EMA(50)
            if close[i] >= donchian_upper[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume and trend confirmation
            if volume[i] > volume_threshold[i]:
                # Bullish breakout: price breaks above upper band with uptrend
                if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower band with downtrend
                elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals