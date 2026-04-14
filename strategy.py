#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1d EMA Trend and Volume Confirmation
# Uses Donchian channel (20-period high/low) from 12h for breakout entries
# 1d EMA (50) provides trend direction filter to trade with higher timeframe trend
# Volume confirmation (>1.3x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of daily trend
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and Donchian calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = price > ema_50_1d_aligned[i]
        price_below_ema = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and above EMA
            if price > donchian_high[i] and vol > 1.3 * avg_vol[i] and price_above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and below EMA
            elif price < donchian_low[i] and vol > 1.3 * avg_vol[i] and price_below_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (reversal) or crosses below EMA
            if price < donchian_low[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (reversal) or crosses above EMA
            if price > donchian_high[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0