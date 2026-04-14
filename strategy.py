#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout with Weekly EMA Filter and Volume Confirmation
# Uses Donchian channel breakout (20-period) from 1d for entry signals
# Weekly EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of weekly trend
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian Channel (20-period) on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian Channel
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of weekly EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume filter and above weekly EMA
            if price > upper_channel[i] and vol > 1.5 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with volume filter and below weekly EMA
            elif price < lower_channel[i] and vol > 1.5 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian (reversal) or below weekly EMA
            if price < lower_channel[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian (reversal) or above weekly EMA
            if price > upper_channel[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0