#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Uses Donchian(20) breakout on 12h as primary signal - captures breakouts in both bull/bear markets
# 1d EMA (50) provides trend filter to avoid counter-trend trades
# Volume spike (>1.5x 20-period average) confirms breakout strength
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over last 20 periods
    high_series = pd.Series(high_12h)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    
    # Lower band: lowest low over last 20 periods
    low_series = pd.Series(low_12h)
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # Volume spike detection (20-period average)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with uptrend filter and volume confirmation
            if price > upper_band_aligned[i] and above_ema and volume_spike:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian band with downtrend filter and volume confirmation
            elif price < lower_band_aligned[i] and not above_ema and volume_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to mid-point of Donchian channel or trend changes
            mid_point = (upper_band_aligned[i] + lower_band_aligned[i]) / 2.0
            if price < mid_point or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to mid-point of Donchian channel or trend changes
            mid_point = (upper_band_aligned[i] + lower_band_aligned[i]) / 2.0
            if price > mid_point or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0