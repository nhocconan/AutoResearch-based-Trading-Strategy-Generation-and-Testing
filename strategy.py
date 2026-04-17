#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h volume confirmation and 1d EMA200 trend filter.
Long when price breaks above 4h Donchian(20) high with volume spike AND price > 1d EMA200 (bullish bias).
Short when price breaks below 4h Donchian(20) low with volume spike AND price < 1d EMA200 (bearish bias).
Exit on opposite Donchian breakout or when price crosses 1d EMA200.
Uses 4h for structure and volume filter, 1d for trend alignment, 1h for precise entry timing.
Target: 60-150 total trades over 4 years (15-37/year). Donchian breakouts capture momentum,
volume confirmation reduces false breakouts, 1d EMA200 filter avoids counter-trend trades in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and volume average
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donch_high_20 = high_4h_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume 20-period average for spike detection
    volume_4h_series = pd.Series(volume_4h)
    vol_ma_20 = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all HTF indicators to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or \
           np.isnan(vol_ma_aligned[i]) or np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_ma = vol_ma_aligned[i]
        ema200 = ema200_aligned[i]
        
        # Volume spike: current 1h volume > 1.5x 4h volume MA (adjust for timeframe difference)
        vol_spike = vol > (vol_ma * 1.5 / 4)  # 4h has 4x the bars of 1h
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume spike AND price > 1d EMA200
            if price > donch_high and vol_spike and price > ema200:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume spike AND price < 1d EMA200
            elif price < donch_low and vol_spike and price < ema200:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low OR price crosses below 1d EMA200
            if price < donch_low or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high OR price crosses above 1d EMA200
            if price > donch_high or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_VolumeSpike_1dEMA200_Trend"
timeframe = "1h"
leverage = 1.0