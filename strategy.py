#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation
# Donchian breakouts capture momentum moves; weekly pivot (from 1w) defines the major trend regime:
#   - Price above weekly pivot = bull regime (only take longs)
#   - Price below weekly pivot = bear regime (only take shorts)
# Volume confirmation filters false breakouts. Works in both bull/bear via regime filter.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.

name = "6h_Donchian20_WeeklyPivot_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly and daily calculations
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3) from previous week
    prev_weekly_high = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
    prev_weekly_low = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
    prev_weekly_close = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) channels on 6h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    start_idx = max(20, 14, 20)  # warmup for Donchian, ATR, volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Price drops below weekly pivot (regime change)
            # 3. Price drops below Donchian low (breakout failed)
            if (curr_low <= stop_price or
                curr_close < curr_weekly_pivot or
                curr_close < curr_donchian_low):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Price rises above weekly pivot (regime change)
            # 3. Price rises above Donchian high (breakout failed)
            if (curr_high >= stop_price or
                curr_close > curr_weekly_pivot or
                curr_close > curr_donchian_high):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter when volume confirms breakout strength
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Donchian high AND above weekly pivot (bull regime)
            if (curr_close > curr_donchian_high and
                curr_close > curr_weekly_pivot):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: price breaks below Donchian low AND below weekly pivot (bear regime)
            elif (curr_close < curr_donchian_low and
                  curr_close < curr_weekly_pivot):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals