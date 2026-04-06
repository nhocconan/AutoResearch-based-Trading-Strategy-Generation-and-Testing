#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w SMA50 trend filter + Donchian(20) breakout
# Long when CHOP > 61.8 (range) AND price > Donchian upper band AND weekly SMA50 rising
# Short when CHOP > 61.8 (range) AND price < Donchian lower band AND weekly SMA50 falling
# Exit when price returns to Donchian midpoint or CHOP < 38.2 (trending)
# Uses 1d timeframe to target 30-100 total trades over 4 years
# Works in both bull/bear markets by combining range-bound mean reversion with trend filter

name = "1d_chop_donchian_1w_sma50_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Choppiness Index (14-period) - measures ranging vs trending
    # High values (>61.8) indicate ranging/choppy market (good for mean reversion)
    # Low values (<38.2) indicate trending market
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    
    atr = np.array(atr_list)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Weekly SMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean()
    sma50_prev = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().shift(1)
    sma50_rising = sma50 > sma50_prev
    sma50_falling = sma50 < sma50_prev
    sma50_aligned = align_htf_to_ltf(prices, df_1w, sma50.values)
    sma50_rising_aligned = align_htf_to_ltf(prices, df_1w, sma50_rising.values)
    sma50_falling_aligned = align_htf_to_ltf(prices, df_1w, sma50_falling.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(chop[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(sma50_aligned[i]) or np.isnan(sma50_rising_aligned[i]) or np.isnan(sma50_falling_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to midpoint OR market starts trending
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (CHOP > 61.8) with Donchian breakout
            # Long: price breaks above Donchian upper band in ranging market + weekly SMA50 rising
            if (chop[i] > 61.8 and close[i] > donch_high[i] and sma50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band in ranging market + weekly SMA50 falling
            elif (chop[i] > 61.8 and close[i] < donch_low[i] and sma50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals