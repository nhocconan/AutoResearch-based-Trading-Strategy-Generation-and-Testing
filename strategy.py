#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Long: Close breaks above Donchian upper AND 12h EMA50 > EMA200 (bullish trend) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian lower AND 12h EMA50 < EMA200 (bearish trend) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or EMA trend flips or volume drops.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure; 12h EMA crossover filters for trending markets only; volume confirmation
# reduces false breakouts. Works in bull via long signals and bear via short signals when aligned with trend.

name = "4h_Donchian20_12hEMA50_200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate 12h EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bullish = ema_50_val > ema_200_val
        is_bearish = ema_50_val < ema_200_val
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND bullish trend AND volume spike
            if close_val > donchian_upper[i] and is_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND bearish trend AND volume spike
            elif close_val < donchian_lower[i] and is_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR trend turns bearish OR volume drops
            if close_val < donchian_lower[i] or not is_bullish or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR trend turns bullish OR volume drops
            if close_val > donchian_upper[i] or is_bullish or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals