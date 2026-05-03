#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Donchian breakouts capture momentum bursts. Filtered by 1w EMA50 trend (bullish if price > EMA50, bearish if price < EMA50)
# and 6h volume > 2.0x 20-period MA to ensure institutional participation.
# Works in bull via long breakouts above upper band and in bear via short breakouts below lower band.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_Donchian20_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from 1w EMA50
        is_bullish_trend = close_val > ema_50_val
        is_bearish_trend = close_val < ema_50_val
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above upper Donchian band AND bullish trend AND volume spike
            if close_val > highest_20[i] and is_bullish_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band AND bearish trend AND volume spike
            elif close_val < lowest_20[i] and is_bearish_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below lower Donchian band OR trend turns bearish OR volume drops
            if close_val < lowest_20[i] or not is_bullish_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above upper Donchian band OR trend turns bullish OR volume drops
            if close_val > highest_20[i] or not is_bearish_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals