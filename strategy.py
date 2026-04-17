#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d EMA trend filter
- Uses 4h Donchian channel (20-period) for structural breakouts
- Volume confirmation: volume > 1.5x 20-period 4h moving average
- Trend filter: 1d EMA50 (price above EMA for longs, below for shorts)
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours
- Fixed position size 0.20 to limit fee churn and manage drawdown
- Designed for low trade frequency (target: 60-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts in uptrends) and bear markets (selling breakdowns in downtrends)
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
    
    # Precompute session hours (08:00-20:00 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel and volume MA
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channel (20-period) - using previous period to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    
    donchian_high = pd.Series(prev_high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA (20-period)
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1h timeframe (primary)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_50 = ema_50_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price breaks above Donchian high + volume spike + price > 1d EMA50
            if price > dch_high and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low + volume spike + price < 1d EMA50
            elif price < dch_low and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < dch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > dch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hVolumeMA_1dEMA50_SessionFilter"
timeframe = "1h"
leverage = 1.0