#!/usr/bin/env python3
"""
1h_4h_DonchianBreakout_TrendFilter_Volume
Hypothesis: Uses 4h Donchian breakouts for trend direction, filtered by 1d EMA trend and volume spikes.
Enters on 1h retracements to the 20-period EMA in the direction of the 4h trend. Designed to work in
bull markets by catching pullbacks in uptrends and in bear markets by shorting bounces in downtrends.
Target: 20-40 trades/year per symbol.
"""

name = "1h_4h_DonchianBreakout_TrendFilter_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # 1h EMA20 for entry timing
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 1h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Determine 4h trend based on Donchian breakout
        trend_4h_up = close[i] > donch_high_aligned[i]
        trend_4h_down = close[i] < donch_low_aligned[i]
        
        if position == 0:
            # Enter long: 4h uptrend + 1d uptrend + price near EMA20 + volume
            if (trend_4h_up and trend_1d_up_aligned[i] > 0.5 and
                close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + 1d downtrend + price near EMA20 + volume
            elif (trend_4h_down and trend_1d_down_aligned[i] > 0.5 and
                  close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                  volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when 4h trend fails or 1d trend changes
            if (trend_4h_down or trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when 4h trend fails or 1d trend changes
            if (trend_4h_up or trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals