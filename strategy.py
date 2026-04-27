#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w Donchian breakout
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean revert),
# CHOP < 38.2 indicates trending market (trend follow).
# Use 1w Donchian breakout for direction in trending regimes,
# and mean reversion at Bollinger Bands in ranging regimes.
# Target: 10-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d Choppiness Index (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of True Range over 14 periods
    tr_sum = atr * 14  # equivalent to tr.rolling(14).sum()
    
    # Max high and min low over 14 periods
    max_high = high_1d.rolling(window=14, min_periods=14).max()
    min_low = low_1d.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: 100 * log10(tr_sum / (max_high - min_low)) / log10(14)
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    # Replace infinities/NaN from zero range with 50 (neutral)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50)
    chop_values = chop.values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 1w Donchian Channel (20-period)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    donch_high = high_1w.rolling(window=20, min_periods=20).max()
    donch_low = low_1w.rolling(window=20, min_periods=20).min()
    donch_high_values = donch_high.values
    donch_low_values = donch_low.values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1w, donch_high_values)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1w, donch_low_values)
    
    # 1d Bollinger Bands (20, 2) for mean reversion in ranging markets
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_values)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(donch_high_1d_aligned[i]) or 
            np.isnan(donch_low_1d_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        if chop_val > 61.8:
            # Ranging market: mean reversion at Bollinger Bands
            if close[i] <= lower_bb_1d_aligned[i] and position != 1:
                signals[i] = 0.25
                position = 1
            elif close[i] >= upper_bb_1d_aligned[i] and position != -1:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position or exit if at mean
                if position == 1 and close[i] >= sma_20.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] <= sma_20.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        elif chop_val < 38.2:
            # Trending market: Donchian breakout
            if close[i] > donch_high_1d_aligned[i] and position != 1:
                signals[i] = 0.25
                position = 1
            elif close[i] < donch_low_1d_aligned[i] and position != -1:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Neutral chop (38.2-61.8): hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_ChopRegime_DonchianBB_1w"
timeframe = "1d"
leverage = 1.0