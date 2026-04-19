#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Volume_Squeeze"
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
    
    # Get 1d data for Camarilla levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # R2 = Close + (High - Low) * 1.1/6
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # S2 = Close - (High - Low) * 1.1/6
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    r2 = close_1d + range_1d * 1.1 / 6
    r3 = close_1d + range_1d * 1.1 / 4
    r4 = close_1d + range_1d * 1.1 / 2
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Band width for squeeze detection (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < 0.8 * bb_width_ma_50  # Bollinger squeeze
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(squeeze_condition[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        squeeze = squeeze_condition[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above S1 with volume and volatility squeeze
            if price > s1_4h[i] and volume_confirmed and squeeze:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume and volatility squeeze
            elif price < r1_4h[i] and volume_confirmed and squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price reaches R3 or closes below S1
            if price >= r3_4h[i] or price < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price reaches S3 or closes above R1
            if price <= s3_4h[i] or price > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals