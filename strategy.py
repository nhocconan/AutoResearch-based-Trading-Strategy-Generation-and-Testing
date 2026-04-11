#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long strategy with volume confirmation and choppiness regime filter
# - Long: price touches Camarilla L3 support (1d) + volume > 1.5x 20-period avg + CHOP > 61.8 (range regime)
# - Exit: price reaches Camarilla L4 level or opposite pivot (H3)
# - Uses 12h timeframe to reduce trade frequency, targeting 50-150 total trades over 4 years
# - Works in both bull and bear markets by mean-reverting in range regimes (CHOP > 61.8)
# - Volume confirmation ensures breakout validity, reducing false signals

name = "12h_1d_camarilla_long_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Calculated from previous day's range: R = High - Low
    # H4 = Close + R * 1.1/2
    # H3 = Close + R * 1.1/4
    # H2 = Close + R * 1.1/6
    # H1 = Close + R * 1.1/12
    # L1 = Close - R * 1.1/12
    # L2 = Close - R * 1.1/6
    # L3 = Close - R * 1.1/4
    # L4 = Close - R * 1.1/2
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first bar where shift creates NaN
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    R = prev_high - prev_low
    H4 = prev_close + R * 1.1 / 2
    H3 = prev_close + R * 1.1 / 4
    H2 = prev_close + R * 1.1 / 6
    H1 = prev_close + R * 1.1 / 12
    L1 = prev_close - R * 1.1 / 12
    L2 = prev_close - R * 1.1 / 6
    L3 = prev_close - R * 1.1 / 4
    L4 = prev_close - R * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low))) over n periods
    # We'll use a simplified version: CHOP > 61.8 indicates range-bound market
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = (range_14 != 0) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / (np.log10(14) * range_14[mask]))
    chop[~mask] = 50  # neutral value when calculation invalid
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        H4 = H4_aligned[i]
        L4 = L4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Regime filter: CHOP > 61.8 indicates range-bound market (good for mean reversion)
        chop_range = chop[i] > 61.8
        
        # Entry conditions (long only)
        enter_long = False
        
        # Long entry: price touches L3 support, volume confirmation, range regime
        if close_price <= L3 and vol_confirm and chop_range:
            enter_long = True
        
        # Exit conditions
        exit_long = False
        
        if position == 1:
            # Exit long if price reaches L4 (extreme support) or H3 (resistance)
            exit_long = close_price <= L4 or close_price >= H3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else 0.0
    
    return signals