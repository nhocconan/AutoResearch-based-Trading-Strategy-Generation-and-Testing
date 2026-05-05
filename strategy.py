#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 1d Donchian upper (20-period high) AND weekly Camarilla S3/R3 shows bullish bias (close > weekly midpoint) AND volume > 2.0 * avg_volume(50) on 6h
# Short when price breaks below 1d Donchian lower (20-period low) AND weekly Camarilla S3/R3 shows bearish bias (close < weekly midpoint) AND volume > 2.0 * avg_volume(50) on 6h
# Exit when price crosses back through the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Donchian provides strong structure that works in both bull and bear markets
# Weekly Camarilla filter ensures we trade with the higher timeframe bias, reducing whipsaw
# Volume confirmation (2.0x) validates breakout strength with strict threshold to avoid overtrading

name = "6h_1dDonchian20_1wCamarillaBias_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_rolling_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = high_rolling_max
    donchian_lower_1d = low_rolling_min
    donchian_mid_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Align 1d Donchian to 6h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # Get 1w data ONCE before loop for Camarilla bias calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_low_1w = high_1w - low_1w
    camarilla_r3_1w = close_1w + 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_s3_1w = close_1w - 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_mid_1w = (camarilla_r3_1w + camarilla_s3_1w) / 2.0
    
    # Align 1w Camarilla to 6h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1w, camarilla_mid_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 50-period average volume on 6h
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * avg_volume_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(avg_volume_50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper, weekly Camarilla shows bullish bias (close > weekly midpoint), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                close_1w[-1] > camarilla_mid_aligned[i] and  # Use latest weekly close for bias
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower, weekly Camarilla shows bearish bias (close < weekly midpoint), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  close_1w[-1] < camarilla_mid_aligned[i] and  # Use latest weekly close for bias
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals