#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Channel Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Donchian breakouts from 12h highs/lows provide clear trend signals.
# Volume confirmation filters out false breakouts. Trend filter (1d EMA50) ensures
# alignment with higher timeframe direction. Works in bull/bear as Donchian adapts to volatility.
# Targets 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.

name = "12h_donchian20_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume confirmation and trend alignment
            if vol_spike[i]:
                # Buy breakout above Donchian high with bullish trend
                if close[i] > highest_high[i] and (i == 100 or close[i-1] <= highest_high[i-1]) and close[i] > ema50_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell breakout below Donchian low with bearish trend
                elif close[i] < lowest_low[i] and (i == 100 or close[i-1] >= lowest_low[i-1]) and close[i] < ema50_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals