#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d KAMA trend filter and volume spike confirmation.
# Williams Alligator (Jaw/Teeth/Lips) identifies trend presence via SMAs.
# Trend: Jaw (13), Teeth (8), Lips (5) - aligned = trending, tangled = ranging.
# KAMA on 1d filters for adaptive trend direction, reducing whipsaw.
# Volume spike confirms breakout strength.
# Designed for low trade frequency (15-40/year) to minimize fee drag.
name = "4h_WilliamsAlligator_1dKAMA_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2.0
    median_series = pd.Series(median_price)
    jaw = median_series.rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = median_series.rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = median_series.rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # KAMA on 1d close
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency Ratio
    change = abs(close_1d.diff(10)).values
    volatility = close_1d.diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d.iloc[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, reverse = downtrend
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up AND price above KAMA AND volume spike
            if alligator_long and price_above_kama and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND price below KAMA AND volume spike
            elif alligator_short and price_below_kama and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator loses alignment OR price crosses below KAMA
            if not alligator_long or not price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator loses alignment OR price crosses above KAMA
            if not alligator_short or not price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals