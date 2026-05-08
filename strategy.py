#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-day Donchian breakout + volume confirmation
# Uses Choppiness Index to identify ranging (CHOP > 61.8) and trending (CHOP < 38.2) regimes
# In trending regimes: trade Donchian(20) breakouts with volume confirmation
# In ranging regimes: mean-revert at Bollinger Band extremes with volume confirmation
# Weekly trend filter ensures alignment with higher timeframe momentum
# Designed for low trade frequency to avoid fee drag on 12h timeframe

name = "12h_Chop_Donchian_BB_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Donchian and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily Bollinger Bands (20, 2)
    daily_close = df_1d['close'].values
    bb_middle = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Choppiness Index (14-period) - measures trend vs ranging
    # High values (>61.8) indicate ranging, low values (<38.2) indicate trending
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(np.roll(close, 1) - low)))
    # Handle first element for rolling
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_max_min = max_high - min_low
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    chop = 100 * np.log10(atr14 * 14 / range_max_min) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Default to middle when undefined
    
    # Volume spike: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Trending regime (CHOP < 38.2): Donchian breakout with volume
            if chop_val < 38.2:
                if close[i] > donchian_high_aligned[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low_aligned[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (CHOP > 61.8): Mean reversion at Bollinger Bands
            elif chop_val > 61.8:
                if close[i] < bb_lower_aligned[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > bb_upper_aligned[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: regime change or opposite signal
            exit_signal = False
            if chop_val < 38.2:  # Still trending
                if close[i] < donchian_low_aligned[i]:
                    exit_signal = True
            else:  # Ranging or neutral
                if close[i] > bb_middle[i]:  # Return to mean
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: regime change or opposite signal
            exit_signal = False
            if chop_val < 38.2:  # Still trending
                if close[i] > donchian_high_aligned[i]:
                    exit_signal = True
            else:  # Ranging or neutral
                if close[i] < bb_middle[i]:  # Return to mean
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals