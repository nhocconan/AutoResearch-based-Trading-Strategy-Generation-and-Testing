#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h Choppiness regime filter and volume confirmation
# Donchian(20) provides clear breakout levels. Chop(12h) > 61.8 filters for ranging markets to avoid false breakouts.
# Volume confirmation ensures breakouts have conviction. Target: 25-40 trades/year to stay under 400 total 4h trades.
# Works in bull via upward breakouts, bear via downward breakouts, avoids choppy markets.
name = "4h_DonchianBreakout_Chop12h_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Chop filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_12h = max_high_12h - min_low_12h
    
    # Chop = 100 * log10(sum_tr / range) / log10(14)
    chop_12h = 100 * np.log10(sum_tr_12h / range_12h) / np.log10(14)
    chop_12h[range_12h == 0] = 50  # avoid division by zero
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4h ATR for position sizing
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period average volume for confirmation
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(chop_12h_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(avg_vol[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.3x average volume
        volume_filter = volume[i] > 1.3 * avg_vol[i]
        
        # Chop filter: avoid ranging markets (Chop > 61.8 = range)
        chop_filter = chop_12h_aligned[i] <= 61.8
        
        if position == 0:
            # Long: Breakout above Donchian high + volume + trending regime
            if price > donch_high[i] and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low + volume + trending regime
            elif price < donch_low[i] and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Breakdown below Donchian low or ATR stop
            if price < donch_low[i] or price < high[i] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Breakout above Donchian high or ATR stop
            if price > donch_high[i] or price < low[i] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals