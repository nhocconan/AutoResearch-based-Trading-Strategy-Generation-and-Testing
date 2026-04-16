#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot filter and volume confirmation.
# Long when price breaks above 20-period 6h Donchian high AND price > 1d weekly pivot R1 AND volume > 1.5x 20-period 6h average volume.
# Short when price breaks below 20-period 6h Donchian low AND price < 1d weekly pivot S1 AND volume > 1.5x 20-period 6h average volume.
# Exit when price crosses the 20-period 6h Donchian midpoint (mean of high/low).
# Uses discrete position size 0.25. Designed to capture breakouts in the direction of the weekly pivot bias.
# Weekly pivot acts as regime filter: R1/S1 from 1d timeframe define bull/bear bias.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's HLC) ===
    df_1d = get_htf_data(prices, '1d')
    # We need to compute weekly pivot from daily data: (Prior Week High + Low + Close) / 3
    # But since we don't have direct weekly data, we approximate using 5-day rolling (1 week ≈ 5 trading days)
    # Weekly high = max of prior 5 daily highs
    # Weekly low = min of prior 5 daily lows
    # Weekly close = close of 5th prior day
    if len(df_1d) >= 5:
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # shift(1) to use prior week
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(df_1d['close']).shift(5).values  # close 5 days ago
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = (2 * weekly_pivot) - weekly_low
        # S1 = (2 * weekly_pivot) - weekly_high
        r1 = (2 * weekly_pivot) - weekly_low
        s1 = (2 * weekly_pivot) - weekly_high
        # Align to 6h timeframe
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    else:
        # Not enough data for weekly pivot
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # enough for 20-period Donchian, 5+20 for weekly pivot, 20 for volume MA
    
    # Track position state and entry price for exit logic
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian midpoint
            if price < donchian_mid[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian midpoint
            if price > donchian_mid[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > weekly R1 AND volume spike
            if (price > highest_high[i]) and (price > r1_aligned[i]) and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND price < weekly S1 AND volume spike
            elif (price < lowest_low[i]) and (price < s1_aligned[i]) and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0