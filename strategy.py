#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R4/S4 breakout on 6h with 1d EMA34 trend filter, volume spike (>2x average), and chop regime filter (choppiness > 61.8 = range -> mean reversion at R3/S3, chop < 38.2 = trend -> breakout continuation at R4/S4). Designed to work in both bull and bear markets by adapting to regime: in trending markets, capture momentum breakouts; in ranging markets, fade extreme levels for mean reversion. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and chop regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (PP * log10(n))) / log10(n)
    # We'll use a rolling window approach
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr = np.maximum(high_arr - low_arr, 
                       np.maximum(np.abs(high_arr - np.roll(close_arr, 1)), 
                                np.abs(low_arr - np.roll(close_arr, 1))))
        tr[0] = high_arr[0] - low_arr[0]
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for Camarilla, 34 for EMA, 14 for ATR, 20 for volume, 14 for chop)
    start_idx = max(20, 34, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Need previous period's OHLC for Camarilla levels
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3, S3 (mean reversion levels) and R4, S4 (breakout levels)
        r3 = prev_close + range_val * 1.25 / 2
        s3 = prev_close - range_val * 1.25 / 2
        r4 = prev_close + range_val * 1.5 / 2
        s4 = prev_close - range_val * 1.5 / 2
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        chop_val = chop_1d_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4) or \
           np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume spike: current volume > 2x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        # Regime determination based on chop
        is_ranging = chop_val > 61.8  # Choppy/ranging market
        is_trending = chop_val < 38.2  # Trending market
        
        # Mean reversion logic (in ranging markets): fade at R3/S3
        mean_revert_long = is_ranging and (close_val < s3) and volume_spike
        mean_revert_short = is_ranging and (close_val > r3) and volume_spike
        
        # Breakout continuation logic (in trending markets): break at R4/S4
        breakout_long = is_trending and (close_val > r4) and (close_val > ema_val) and volume_spike
        breakout_short = is_trending and (close_val < s4) and (close_val < ema_val) and volume_spike
        
        # Exit logic: 
        # - For mean reversion: exit when price crosses mid-point (prev_close)
        # - For breakout: exit when trend reverses (price crosses 1d EMA34) or opposite signal
        exit_mean_revert_long = is_ranging and (close_val > prev_close)
        exit_mean_revert_short = is_ranging and (close_val < prev_close)
        exit_breakout = (close_val < ema_val) if position == 1 else (close_val > ema_val)
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if (mean_revert_long or breakout_long) and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif (mean_revert_short or breakout_short) and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        # Exit logic
        elif position == 1 and (exit_mean_revert_long or exit_breakout):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (exit_mean_revert_short or exit_breakout):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "6h"
leverage = 1.0