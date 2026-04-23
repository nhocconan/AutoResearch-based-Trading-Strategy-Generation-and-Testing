#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Daily Donchian channel breakouts capture strong momentum moves
- Only trade breakouts aligned with weekly EMA(50) trend to avoid counter-trend whipsaws
- Volume confirmation (> 1.5x 20-period average) ensures breakout has conviction
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading with the weekly trend
- Donchian levels adapt to volatility, providing dynamic support/resistance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Use rolling window on daily data - need to resample to daily first for proper calculation
    # But we can approximate using the prices DataFrame which is already at primary timeframe
    # For 1d primary timeframe, we can calculate directly
    if len(prices) >= 20:
        # For 1d timeframe, calculate rolling max/min of high/low
        high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above 20-day high (Donchian upper band) with volume
        # Short: price breaks below 20-day low (Donchian lower band) with volume
        price_above_donchian_high = close[i] > high_max[i]
        price_below_donchian_low = close[i] < low_min[i]
        
        # Trend filter: price > weekly EMA for long, price < weekly EMA for short
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, uptrend, volume spike
            long_signal = (price_above_donchian_high and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low, downtrend, volume spike
            short_signal = (price_below_donchian_low and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian band break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or trend turns down
                if (price_below_donchian_low or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend turns up
                if (price_above_donchian_high or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0