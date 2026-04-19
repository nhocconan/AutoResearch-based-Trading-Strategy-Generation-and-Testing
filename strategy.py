#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# Only enters long when price breaks above 12h Donchian high + price > 1d EMA200 + volume > 1.5x average
# Only enters short when price breaks below 12h Donchian low + price < 1d EMA200 + volume > 1.5x average
# Uses ATR-based stop loss to limit drawdown
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via breakouts above EMA200, in bear via breakdowns below EMA200
name = "12h_DonchianBreakout_EMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: Price breaks above Donchian high + price > 1d EMA200 + volume
            if price > donch_high[i] and price > ema200_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + price < 1d EMA200 + volume
            elif price < donch_low[i] and price < ema200_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Donchian low or ATR stop
            if price < donch_low[i] or price < high[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Donchian high or ATR stop
            if price > donch_high[i] or price > low[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals