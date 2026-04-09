#!/usr/bin/env python3
# 4h_donchian_12h_trend_volume_v1
# Hypothesis: 4h Donchian breakout with 12h trend filter (EMA50) and volume confirmation.
# Works in bull/bear: 12h EMA50 defines medium-term trend; Donchian(20) breakouts capture momentum;
# volume confirms institutional participation. Target: 19-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian midpoint OR trend turns bearish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian midpoint OR trend turns bullish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above Donchian high with bullish trend
                if close[i] > highest_high[i] and close[i] > ema50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with bearish trend
                elif close[i] < lowest_low[i] and close[i] < ema50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals