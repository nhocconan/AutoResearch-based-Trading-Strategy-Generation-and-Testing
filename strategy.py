#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Channel Breakout with 1d Trend Filter and Volume Spike
# Uses 12h Donchian channels (20-period) for breakout entries
# 1d EMA (50) provides daily trend direction to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drift

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d EMA data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian Channel (20-period) on 12h
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 1d EMA and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_donch_aligned[i]) or 
            np.isnan(lower_donch_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        trend_up = price > ema_50_1d_aligned[i]
        trend_down = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume filter and uptrend
            if price > upper_donch_aligned[i] and vol > 1.5 * avg_vol[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with volume filter and downtrend
            elif price < lower_donch_aligned[i] and vol > 1.5 * avg_vol[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian (mean reversion)
            if price < lower_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian (mean reversion)
            if price > upper_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0