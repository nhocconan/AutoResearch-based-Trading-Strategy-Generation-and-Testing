#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d CAMARILLA PIVOT LEVELS with volume confirmation
# Uses Choppiness Index to determine market regime (trending vs ranging) and trades accordingly:
# - In trending markets (CHOP < 38.2): breakout trades at 1d CAMARILLA H3/L3 levels
# - In ranging markets (CHOP > 61.8): mean-reversion trades at 1d CAMARILLA H4/L4 levels
# Volume filter ensures only significant moves are traded. Designed for low trade frequency
# (~25-35 trades/year) to avoid fee drag while capturing moves in both bull and bear markets.
# Entry/exit logic uses price closes only, no look-ahead, with proper multi-timeframe alignment.
name = "4h_Chop_Camarilla_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Calculate Choppiness Index on 4h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = tr1[0]
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.full_like(close, 50.0)  # Default to neutral
    mask = (range_hl > 0) & (~np.isnan(tr_sum))
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # 2. Get 1d data for CAMARILLA pivot levels (calculate ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate typical price for CAMARILLA
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    # CAMARILLA levels based on previous day's typical price
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    
    # Use previous day's values to avoid look-ahead
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.125 * (prev_high - prev_low)
    L3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Align to 4h timeframe with proper delay (wait for daily close)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    
    # 3. Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime and trade accordingly
            if chop[i] < 38.2:  # Trending market - breakout
                # Long breakout above H3
                if close[i] > H3_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below L3
                elif close[i] < L3_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop[i] > 61.8:  # Ranging market - mean reversion
                # Long near L4 (support)
                if close[i] <= L4_aligned[i] * 1.002 and volume_filter[i]:  # Small buffer
                    signals[i] = 0.25
                    position = 1
                # Short near H4 (resistance)
                elif close[i] >= H4_aligned[i] * 0.998 and volume_filter[i]:  # Small buffer
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long position management
            if chop[i] < 38.2:  # Trending - trail with H3 break
                if close[i] < H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging or choppy - mean reversion target at H3
                if close[i] >= H3_aligned[i] * 0.995:  # Near H3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            if chop[i] < 38.2:  # Trending - trail with L3 break
                if close[i] > L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging or choppy - mean reversion target at L3
                if close[i] <= L3_aligned[i] * 1.005:  # Near L3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals