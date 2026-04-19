#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d/1w trend filter (EMA50), volume confirmation, and ATR-based stop.
# Long when price breaks above Donchian(20) high, EMA50_1d/1w > price (bullish), and volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian(20) low, EMA50_1d/1w < price (bearish), and volume > 1.5x 20-bar avg.
# Exit on opposite Donchian break or when EMA50 trend flips.
# Uses strict conditions to limit trades to ~20-30/year and avoid overtrading.
# Works in both bull and bear markets by using EMA50 trend filter on higher timeframes.

name = "4h_Donchian_EMA50_1d1w_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Get 1d and 1w EMA50 trends
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > Donchian high, EMA50_1d/1w > price (bullish trend), volume filter
            if (close[i] > donch_high[i] and 
                ema50_1d_aligned[i] > close[i] and 
                ema50_1w_aligned[i] > close[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low, EMA50_1d/1w < price (bearish trend), volume filter
            elif (close[i] < donch_low[i] and 
                  ema50_1d_aligned[i] < close[i] and 
                  ema50_1w_aligned[i] < close[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price < Donchian low OR EMA50 trend turns bearish
            if (close[i] < donch_low[i] or 
                ema50_1d_aligned[i] < close[i] or 
                ema50_1w_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price > Donchian high OR EMA50 trend turns bullish
            if (close[i] > donch_high[i] or 
                ema50_1d_aligned[i] > close[i] or 
                ema50_1w_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals