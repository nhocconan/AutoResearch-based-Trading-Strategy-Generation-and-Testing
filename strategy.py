#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above daily Donchian high(20) AND weekly EMA50 is rising AND 1d volume > 1.5x 20-day average.
# Short when price breaks below daily Donchian low(20) AND weekly EMA50 is falling AND 1d volume > 1.5x 20-day average.
# Uses weekly trend for direction and daily volume for momentum confirmation to reduce false breakouts.
# Designed for 1d timeframe with target of 8-20 trades/year (32-80 total over 4 years).
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
name = "1d_Donchian20_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_1w > np.roll(ema50_1w, 1)
    ema50_falling = ema50_1w < np.roll(ema50_1w, 1)
    ema50_rising[0] = False
    ema50_falling[0] = False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_confirm = df_1d['volume'].values > (1.5 * vol_ma_20)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian high, weekly EMA50 rising, volume confirmation
            long_condition = (close[i] > donchian_high_aligned[i]) and ema50_rising_aligned[i] and vol_confirm_aligned[i]
            # Short condition: break below Donchian low, weekly EMA50 falling, volume confirmation
            short_condition = (close[i] < donchian_low_aligned[i]) and ema50_falling_aligned[i] and vol_confirm_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly EMA50 turns falling
            if (close[i] < donchian_low_aligned[i]) or (~ema50_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly EMA50 turns rising
            if (close[i] > donchian_high_aligned[i]) or ema50_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals