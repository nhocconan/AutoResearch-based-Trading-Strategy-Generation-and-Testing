#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band squeeze + 1w Donchian breakout direction
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1w Donchian(20) breakout provides the directional bias for the squeeze breakout
# In squeeze (BBW < 20th percentile): await breakout in direction of 1w Donchian trend
# Outside squeeze: stay flat to avoid whipsaws in choppy markets
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakouts capture explosive moves regardless of trend direction

name = "6h_1d_1w_bb_squeeze_donchian_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_s_1d = pd.Series(close_1d)
    basis_1d = close_s_1d.rolling(window=20, min_periods=20).mean()
    dev_1d = close_s_1d.rolling(window=20, min_periods=20).std()
    upper_1d = basis_1d + 2.0 * dev_1d
    lower_1d = basis_1d - 2.0 * dev_1d
    
    # Bollinger Band Width
    bbw_1d = (upper_1d - lower_1d) / basis_1d
    
    # Calculate 20th percentile of BBW for squeeze detection (using expanding window)
    def rolling_percentile(values, window, percentile):
        result = np.full(len(values), np.nan)
        for i in range(len(values)):
            if i < window - 1:
                result[i] = np.nan
            else:
                window_data = values[max(0, i-window+1):i+1]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) >= 10:  # minimum samples for percentile
                    result[i] = np.percentile(valid_data, percentile)
                else:
                    result[i] = np.nan
        return result
    
    bbw_percentile_20 = rolling_percentile(bbw_1d.values, 50, 20)  # 20th percentile over 50 periods (~1 year)
    squeeze_condition = bbw_1d.values < bbw_percentile_20
    
    # Load 1w data for Donchian direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian Channels (20)
    def donchian_channels(high, low, window=20):
        upper = np.full(len(high), np.nan)
        lower = np.full(len(low), np.nan)
        for i in range(len(high)):
            if i < window - 1:
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = np.max(high[i-window+1:i+1])
                lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    donch_upper_1w, donch_lower_1w = donchian_channels(high_1w, low_1w, 20)
    
    # Donchian breakout direction
    donch_long_break = high_1w > donch_upper_1w  # break above upper channel
    donch_short_break = low_1w < donch_lower_1w  # break below lower channel
    
    # Align 1d indicators to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    bbw_aligned = align_htf_to_ltf(prices, df_1d, bbw_1d.values)
    
    # Align 1w indicators to 6h timeframe
    donch_long_aligned = align_htf_to_ltf(prices, df_1w, donch_long_break)
    donch_short_aligned = align_htf_to_ltf(prices, df_1w, donch_short_break)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper_1w)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(squeeze_aligned[i]) or np.isnan(donch_long_aligned[i]) or 
            np.isnan(donch_short_aligned[i]) or np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if squeeze ends or opposite breakout occurs
            if not squeeze_aligned[i] or donch_short_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if squeeze ends or opposite breakout occurs
            if not squeeze_aligned[i] or donch_long_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only during squeeze in direction of 1w Donchian breakout
            if squeeze_aligned[i]:
                if donch_long_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif donch_short_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals