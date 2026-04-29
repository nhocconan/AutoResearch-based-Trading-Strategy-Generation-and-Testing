#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R extreme filter and volume confirmation
# Long when price breaks above 6h Donchian upper band AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-bar avg
# Short when price breaks below 6h Donchian lower band AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-bar avg
# Exit when price crosses 6h Donchian midpoint (mean reversion to mean)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to avoid overtrading.
# Williams %R provides mean reversion edge in ranging markets while Donchian breakout captures trends.
# Works in bull markets by buying oversold breakouts and in bear markets by selling overbought breakdowns.

name = "6h_Donchian20_WilliamsR_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Donchian and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_williams_r = williams_r_aligned[i]
        curr_highest_high = highest_high_20[i]
        curr_lowest_low = lowest_low_20[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses Donchian midpoint (mean reversion)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses Donchian midpoint (mean reversion)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND Williams %R < -80 (oversold) AND volume confirmation
            if curr_close > curr_highest_high and curr_williams_r < -80 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND Williams %R > -20 (overbought) AND volume confirmation
            elif curr_close < curr_lowest_low and curr_williams_r > -20 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals