#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# Camarilla pivots from 1d provide intraday support/resistance levels that work in ranging and trending markets
# Volume confirmation (current 12h volume > 1.3x 20-period average) filters false breakouts
# Choppiness regime (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) adapts strategy to market conditions
# In ranging markets (CHOP > 61.8): mean reversion at H3/L3 levels
# In trending markets (CHOP < 38.2): breakout continuation at H4/L4 levels
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    h4 = pivot + (range_hl * 1.1 / 2)
    h3 = pivot + (range_hl * 1.1 / 4)
    h2 = pivot + (range_hl * 1.1 / 6)
    h1 = pivot + (range_hl * 1.1 / 12)
    l1 = pivot - (range_hl * 1.1 / 12)
    l2 = pivot - (range_hl * 1.1 / 6)
    l3 = pivot - (range_hl * 1.1 / 4)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Align Camarilla levels and Choppiness to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x average 12h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        chop_value = chop_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions based on regime
            if chop_value > 61.8:  # Ranging market: mean reversion at L3
                if close[i] < l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending market: trail with L4 break
                if close[i] < l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if chop_value > 61.8:  # Ranging market: mean reversion at H3
                if close[i] > h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending market: trail with H4 break
                if close[i] > h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime and volume confirmation
            if chop_value > 61.8:  # Ranging market: mean reversion
                # Long at L3, Short at H3
                if close[i] < l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # Trending market: breakout continuation
                # Long at H4 breakout, Short at L4 breakout
                if close[i] > h4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals