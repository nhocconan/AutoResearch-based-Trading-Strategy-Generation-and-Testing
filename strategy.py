#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action with 1d Bollinger Band squeeze breakout and volume confirmation.
# Long: Price closes above upper Bollinger Band + Bollinger Band width at 20-day low + volume > 1.5x average.
# Short: Price closes below lower Bollinger Band + Bollinger Band width at 20-day low + volume > 1.5x average.
# Uses Bollinger Band squeeze (low volatility breakout) as the primary signal with volume confirmation.
# Timeframe: 4h for optimal balance of signal quality and trade frequency.
# Target: 80-180 total trades over 4 years (20-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2) on daily close
    bb_length = 20
    bb_std = 2.0
    
    sma = np.full(len(close_1d), np.nan)
    std_dev = np.full(len(close_1d), np.nan)
    upper_band = np.full(len(close_1d), np.nan)
    lower_band = np.full(len(close_1d), np.nan)
    bb_width = np.full(len(close_1d), np.nan)
    
    for i in range(bb_length - 1, len(close_1d)):
        sma[i] = np.mean(close_1d[i - bb_length + 1:i + 1])
        std_dev[i] = np.std(close_1d[i - bb_length + 1:i + 1])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
        bb_width[i] = (upper_band[i] - lower_band[i]) / sma[i] * 100  # Percentage width
    
    # Calculate 20-period lowest Bollinger Band width (squeeze condition)
    bb_width_lowest = np.full(len(close_1d), np.nan)
    lookback = 20
    for i in range(lookback - 1, len(close_1d)):
        bb_width_lowest[i] = np.min(bb_width[i - lookback + 1:i + 1])
    
    # Squeeze condition: current BB width is at or near the 20-period low
    squeeze_condition = bb_width <= bb_width_lowest * 1.1  # Within 10% of the low
    
    # Average volume (20-period) for volume confirmation on 4h data
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Bollinger Bands and squeeze condition to 4h
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        is_squeeze = squeeze_aligned[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price closes above upper BB during squeeze + volume confirmation
            if (price > upper and is_squeeze and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower BB during squeeze + volume confirmation
            elif (price < lower and is_squeeze and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower Bollinger Band
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Bollinger Band
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Squeeze_Breakout_Volume"
timeframe = "4h"
leverage = 1.0