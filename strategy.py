#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Donchian channel (20-period high/low) provides robust structure for breakouts
- Only trade breakouts in direction of 1w EMA(50) trend to avoid counter-trend whipsaws
- Volume confirmation (> 2.0x 20-period average) ensures breakout has momentum
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels on 1d timeframe (using current prices)
    # Upper channel: 20-period high, Lower channel: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above Donchian upper (resistance) with volume
        # Short: price breaks below Donchian lower (support) with volume
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper, uptrend, volume spike
            long_signal = (price_above_upper and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below lower, downtrend, volume spike
            short_signal = (price_below_lower and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite level break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower or trend turns down
                if (price_below_lower or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper or trend turns up
                if (price_above_upper or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0