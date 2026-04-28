#!/usr/bin/env python3
"""
4h_21HMA_Trend_BollingerBreakout_12hVolume
Hypothesis: Use 21-period Hull Moving Average on 4h for trend direction, breakout from Bollinger Bands on 4h for entry, 
and volume spike on 12h as confirmation. Designed to work in both bull and bear markets by following the trend 
with volatility-based entries and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 12h data
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 30 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 21-period HMA on 4h close
    hma_21_4h = calculate_hma(df_4h['close'].values, 21)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    
    # Calculate Bollinger Bands on 4h (20-period, 2 std dev)
    sma_20_4h = pd.Series(df_4h['close']).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(df_4h['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma_20_4h + (2 * std_20_4h)
    lower_bb_4h = sma_20_4h - (2 * std_20_4h)
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    
    # Calculate volume spike on 12h (current volume > 2.0x 20-period average)
    vol_ma_20_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = df_12h['volume'].values > (vol_ma_20_12h * 2.0)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_4h_aligned[i]) or np.isnan(upper_bb_4h_aligned[i]) or 
            np.isnan(lower_bb_4h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below HMA
        price_above_hma = close[i] > hma_21_4h_aligned[i]
        price_below_hma = close[i] < hma_21_4h_aligned[i]
        
        # Bollinger Band breakout conditions
        breakout_above = close[i] > upper_bb_4h_aligned[i]
        breakout_below = close[i] < lower_bb_4h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike_12h_aligned[i]
        
        # Entry conditions
        long_entry = breakout_above and price_above_hma and vol_spike
        short_entry = breakout_below and price_below_hma and vol_spike
        
        # Exit conditions: opposite Bollinger Band touch
        long_exit = close[i] < lower_bb_4h_aligned[i]
        short_exit = close[i] > upper_bb_4h_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_21HMA_Trend_BollingerBreakout_12hVolume"
timeframe = "4h"
leverage = 1.0