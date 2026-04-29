#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d HMA21 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1d HMA21 AND volume > 1.5x 20-bar avg
# Exit when price retouches Donchian midpoint or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year on 4h.
# Donchian channels provide structural breakout levels with proven edge in trending markets.
# 1d HMA21 filter ensures we only trade with the daily trend, improving win rate in bear markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.

name = "4h_Donchian20_1dHMA21_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data for HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate HMA(21) on 1d data
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half_n, adjust=False, min_periods=half_n).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).ewm(span=sqrt_n, adjust=False, min_periods=sqrt_n).mean().values
    hma_21_1d = wma_sqrt
    # Align HMA21 to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA need 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        hma_21 = hma_21_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d HMA21 AND volume confirmation
            if curr_high > upper and curr_close > hma_21 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d HMA21 AND volume confirmation
            elif curr_low < lower and curr_close < hma_21 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches midpoint or breaks below lower
            if curr_close <= mid or curr_low < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches midpoint or breaks above upper
            if curr_close >= mid or curr_high > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals