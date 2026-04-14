#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from 1-day data with volume spike and choppiness regime filter
# Long when price touches or crosses above Camarilla H4 level AND volume > 2x 20-period average AND 12h choppiness < 61.8 (trending)
# Short when price touches or crosses below Camarilla L4 level AND volume > 2x 20-period average AND 12h choppiness < 61.8 (trending)
# Exit when price crosses back to Camarilla H3/L3 levels or choppiness > 61.8 (range)
# Uses daily pivot levels for structure, volume for confirmation, choppiness for regime
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily typical price for pivot
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on daily data
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    hl_range_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.5 * hl_range_1d
    camarilla_l4_1d = close_1d - 1.5 * hl_range_1d
    camarilla_h3_1d = close_1d + 1.1 * hl_range_1d
    camarilla_l3_1d = close_1d - 1.1 * hl_range_1d
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d.values)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d.values)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d.values)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d.values)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate choppiness index on 12h data (14-period)
    # Chop = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(camarilla_h3_12h[i]) or np.isnan(camarilla_l3_12h[i]) or
            np.isnan(vol_avg[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0  # Require 2x average volume
        
        if position == 0:
            # Long setup: price at/above H4 + volume spike + trending regime (Chop < 61.8)
            if (price >= camarilla_h4_12h[i] and vol > vol_threshold and chop[i] < 61.8):
                position = 1
                signals[i] = position_size
            # Short setup: price at/below L4 + volume spike + trending regime (Chop < 61.8)
            elif (price <= camarilla_l4_12h[i] and vol > vol_threshold and chop[i] < 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below H3 OR chop > 61.8 (range)
            if price < camarilla_h3_12h[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above L3 OR chop > 61.8 (range)
            if price > camarilla_l3_12h[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dVOL_Chop"
timeframe = "12h"
leverage = 1.0