#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining 1w Bollinger Band squeeze (volatility contraction) 
# with 1w Donchian breakout direction and volume confirmation. 
# Bollinger Band Width < 20th percentile indicates low volatility squeeze.
# Breakout direction determined by 1w Donchian(20): long if price > upper band, short if price < lower band.
# Volume confirmation: current 1d volume > 1.5 * 20-period average volume.
# Only trade in low volatility regimes to capture explosive moves after consolidation.
# Works in bull/bear markets: volatility contraction precedes big moves in both directions.
# Uses discrete position sizing 0.25 to target ~10-25 trades/year and minimize fee drag.

name = "1d_1w_bb_squeeze_donchian_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Bollinger Bands (20, 2)
    close_s_1w = pd.Series(close_1w)
    basis_1w = close_s_1w.ewm(span=20, adjust=False, min_periods=20).mean()
    dev_1w = 2 * close_s_1w.ewm(span=20, adjust=False, min_periods=20).std()
    upper_1w = basis_1w + dev_1w
    lower_1w = basis_1w - dev_1w
    bBW_1w = (upper_1w - lower_1w) / basis_1w  # Band Width
    
    # Calculate 20th percentile of BBW for squeeze condition
    def rolling_percentile(arr, window, percentile):
        from scipy.stats import percentileofscore
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                result[i] = percentileofscore(valid_data, arr[i], kind='rank') / 100.0 * 100
        return result
    
    # Approximate percentile using rolling min/max for efficiency
    min_bBW_1w = pd.Series(bBW_1w).rolling(window=50, min_periods=20).min().values
    max_bBW_1w = pd.Series(bBW_1w).rolling(window=50, min_periods=20).max().values
    range_bBW_1w = max_bBW_1w - min_bBW_1w
    # Avoid division by zero
    range_bBW_1w = np.where(range_bBW_1w == 0, 1, range_bBW_1w)
    percent_bBW_1w = 100 * (bBW_1w - min_bBW_1w) / range_bBW_1w
    squeeze_condition = percent_bBW_1w < 20  # BBW in lower 20th percentile
    
    # Calculate 1w Donchian Channel (20)
    def donchian_channel(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    upper_dc_1w, lower_dc_1w = donchian_channel(high_1w, low_1w, 20)
    donchian_breakout_up = close_1w > upper_dc_1w
    donchian_breakout_down = close_1w < lower_dc_1w
    
    # Calculate 1w volume average (20-period)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.5 * vol_ma_1w)
    
    # Align 1w indicators to 1d timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, donchian_breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, donchian_breakout_down)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: squeeze + breakout direction + volume spike
        long_entry = squeeze_aligned[i] and breakout_up_aligned[i] and volume_spike_aligned[i]
        short_entry = squeeze_aligned[i] and breakout_down_aligned[i] and volume_spike_aligned[i]
        
        # Exit conditions: exit when squeeze ends (volatility expands) or opposite signal
        long_exit = not squeeze_aligned[i] or breakout_down_aligned[i]
        short_exit = not squeeze_aligned[i] or breakout_up_aligned[i]
        
        if position == 1:  # Long position
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals