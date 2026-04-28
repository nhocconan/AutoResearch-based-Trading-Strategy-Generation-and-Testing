#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with 12h HMA trend filter and volume confirmation.
# Enter long when price breaks above 1d Donchian upper channel (20-period) with volume > 1.8x 20-bar average and close > 12h HMA21.
# Enter short when price breaks below 1d Donchian lower channel (20-period) with volume > 1.8x average and close < 12h HMA21.
# Exit when price returns to the 1d Donchian midpoint.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Uses 1d Donchian for structure (more stable than lower TF) and 12h HMA21 for trend filter (reduces whipsaws).

name = "4h_Donchian_20_12hHMA21_VolumeConfirm_v1"
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
    
    # Get 1d data for Donchian channel calculation (MTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower channels (20-period high/low)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get 12h data for HMA21 trend filter (MTF trend)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    close_12h = df_12h['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_length)
    wma_full = wma(close_12h, 21)
    wma_diff = 2 * wma_half - wma_full
    # Pad the beginning with NaN to align lengths
    wma_diff_padded = np.full(len(close_12h) - len(wma_diff), np.nan)
    wma_diff_padded = np.concatenate([wma_diff_padded, wma_diff])
    hma_21_12h = wma(wma_diff_padded, sqrt_length)
    # Pad the beginning again for final alignment
    hma_21_12h_padded = np.full(len(close_12h) - len(hma_21_12h), np.nan)
    hma_21_12h = np.concatenate([hma_21_12h_padded, hma_21_12h])
    
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 12h HMA21 bias
        bullish_bias = close[i] > hma_21_12h_aligned[i]
        bearish_bias = close[i] < hma_21_12h_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Exit condition: return to midpoint
        long_exit = close[i] < donchian_mid_aligned[i]
        short_exit = close[i] > donchian_mid_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals