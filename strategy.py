#!/usr/bin/env python3
# 4h_1D_Donchian20_With_1D_TrendFilter_VolumeSpike
# Hypothesis: On 4h timeframe, enter long when price breaks above 4h Donchian(20) high
# with volume confirmation and daily trend filter; enter short when price breaks below
# 4h Donchian(20) low with volume confirmation and daily trend filter. This captures
# breakouts in both bull and bear markets. Uses volume > 2x 20-period average for
# confirmation and only trades in direction of daily EMA(50) trend to avoid counter-trend.
# Exit when price retests the Donchian midpoint or reverses. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Donchian20_With_1D_TrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Donchian channel (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high and low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Donchian midpoint for exit
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align daily EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and Donchian warmup
        # Get values
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high_val) or np.isnan(donch_low_val) or 
            np.isnan(donch_mid_val) or np.isnan(ema50_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume confirmation and above daily EMA50
            if (close_val > donch_high_val and  # Break above Donchian high
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val > ema50_1d_val):  # Only long in daily uptrend
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume confirmation and below daily EMA50
            elif (close_val < donch_low_val and  # Break below Donchian low
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val < ema50_1d_val):  # Only short in daily downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retests Donchian midpoint or breaks below Donchian low
            if close_val <= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retests Donchian midpoint or breaks above Donchian high
            if close_val >= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals