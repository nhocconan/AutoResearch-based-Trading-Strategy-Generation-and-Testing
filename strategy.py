#!/usr/bin/env python3
"""
12h_triple_barrier_v1
Hypothesis: Use 12h price action with 1d trend filter and volume confirmation.
- Long: Price breaks above 12h Donchian high(20) + 1d trend bullish (close > EMA50) + volume > 1.5x avg
- Short: Price breaks below 12h Donchian low(20) + 1d trend bearish (close < EMA50) + volume > 1.5x avg
- Exit: Opposite Donchian break or trend reversal
- Position size: 0.25 to limit drawdown
- Target: 20-40 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_triple_barrier_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend direction
    bullish_trend = close_1d > ema_50
    bearish_trend = close_1d < ema_50
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price breaks below Donchian low OR trend turns bearish
            if low[i] < donchian_low[i] or bearish_trend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above Donchian high OR trend turns bullish
            if high[i] > donchian_high[i] or bullish_trend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: break above Donchian high + bullish trend + volume
            if (high[i] > donchian_high[i] and 
                bullish_trend_aligned[i] > 0.5 and 
                volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian low + bearish trend + volume
            elif (low[i] < donchian_low[i] and 
                  bearish_trend_aligned[i] > 0.5 and 
                  volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals