#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 1D Volume Spike and Choppiness Filter
# Camarilla pivot levels (support/resistance) derived from previous day's OHLC provide high-probability breakout levels.
# Volume spike (>2x average) confirms institutional participation and reduces false breakouts.
# Choppiness index > 61.8 indicates ranging market where mean reversion at pivot levels works best.
# Designed to work in both bull and bear markets by trading mean reversion in ranging conditions and breakouts in trending.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE before loop for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D typical price for pivot calculations
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_arr = typical_price.values
    
    # Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 1.6 * (High - Low) / 2
    # R1 = Close + 1.1 * (High - Low) / 2
    # PP = (High + Low + Close) / 3
    # S1 = Close - 1.1 * (High - Low) / 2
    # S2 = Close - 1.6 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s2 = close_1d - 0.55 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 0.275 * (high_1d - low_1d)
    camarilla_pp = typical_price_arr
    camarilla_r1 = close_1d + 0.275 * (high_1d - low_1d)
    camarilla_r2 = close_1d + 0.55 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # Calculate Choppiness Index on 1D (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum()
    max_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_values = chop_raw.fillna(50).values  # fill NaN with 50 (neutral)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Camarilla and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Choppiness filter: only trade when market is ranging (CHOP > 61.8)
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above S1 with volume filter in ranging market
            if price > s1_aligned[i] and vol > 2.0 * avg_vol[i] and is_ranging:
                position = 1
                signals[i] = position_size
            # Short: price breaks below R1 with volume filter in ranging market
            elif price < r1_aligned[i] and vol > 2.0 * avg_vol[i] and is_ranging:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot point or stops below S1
            if price >= pp_aligned[i] or price <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot point or stops above R1
            if price <= pp_aligned[i] or price >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Chop"
timeframe = "4h"
leverage = 1.0