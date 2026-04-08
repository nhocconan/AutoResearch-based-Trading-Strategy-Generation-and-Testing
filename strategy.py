#!/usr/bin/env python3
# 1h_momentum_breakout_v1
# Hypothesis: 1h momentum breakout with 4h/1d trend filter and volume confirmation.
# Uses 4h EMA40 for trend direction and 1d EMA200 for market regime filter.
# Enters on 1h Donchian breakout (20-period) with volume > 1.5x average.
# Designed to work in both bull and bear markets by following higher timeframe trends.
# Target: 15-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_breakout_v1"
timeframe = "1h"
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
    
    # 4h EMA40 for trend direction - load once before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema40_4h = pd.Series(close_4h).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_4h_aligned = align_htf_to_ltf(prices, df_4h, ema40_4h)
    
    # 1d EMA200 for market regime - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema40_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters
        bullish_regime = close[i] > ema200_1d_aligned[i]
        bearish_regime = close[i] < ema200_1d_aligned[i]
        bullish_trend = close[i] > ema40_4h_aligned[i]
        bearish_trend = close[i] < ema40_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend reversal or price retracement to midline
            if not bullish_trend or close[i] < (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend reversal or price retracement to midline
            if not bearish_trend or close[i] > (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: bullish regime + bullish trend + breakout above Donchian high
                if bullish_regime and bullish_trend and close[i] > highest_high[i] and close[i-1] <= highest_high[i-1]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: bearish regime + bearish trend + breakdown below Donchian low
                elif bearish_regime and bearish_trend and close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1]:
                    position = -1
                    signals[i] = -0.20
    
    return signals