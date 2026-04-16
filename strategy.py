#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above R1 AND volume > 1.8x 20-period 4h average AND 1d Choppiness Index > 61.8 (ranging market).
# Short when price breaks below S1 AND volume > 1.8x 20-period 4h average AND 1d Choppiness Index > 61.8.
# Exit when price crosses the 4h midpoint (R1+S1)/2.
# Uses discrete position size 0.25. Designed to capture mean-reversion bounces at strong intraday levels
# in ranging markets, avoiding trending conditions where breakouts fail.
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla R1/S1 levels (from previous 4h bar) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla R1 and S1 levels
    R1_4h = pivot_4h + (range_4h * 1.1 / 12)  # R1 = pivot + range*1.1/12
    S1_4h = pivot_4h - (range_4h * 1.1 / 12)  # S1 = pivot - range*1.1/12
    midpoint_4h = (R1_4h + S1_4h) / 2         # Exit level
    
    # Align Camarilla levels to 15m timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    
    # === 4h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.8 * vol_ma_4h_aligned)
    
    # === 1d Indicators: Choppiness Index > 61.8 (ranging market filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest High and Lowest Low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ranging = chop_aligned > 61.8  # Chop > 61.8 indicates ranging market
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or np.isnan(midpoint_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ranging[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_ranging = ranging[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < midpoint_4h_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > midpoint_4h_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND ranging market
            if price > R1_4h_aligned[i] and vol_spike and is_ranging:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND volume spike AND ranging market
            elif price < S1_4h_aligned[i] and vol_spike and is_ranging:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_1dChop_V1"
timeframe = "4h"
leverage = 1.0