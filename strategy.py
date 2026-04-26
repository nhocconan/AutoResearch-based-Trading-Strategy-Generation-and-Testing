#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, choppiness regime filter (CHOP > 38.2 = trending), and volume spike confirmation. Uses discrete sizing 0.25 to control trade frequency (~20-30/year). Designed to work in both bull and bear markets via 1d trend filter and regime filter to avoid whipsaws in ranging markets. Volume spike ensures momentum behind breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3, S3, R4, S4 for each 1d bar
    rng = high_1d - low_1d
    r3 = close_1d + 1.125 * rng
    s3 = close_1d - 1.125 * rng
    r4 = close_1d + 1.5 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop_raw = 100 * np.log10(atr * atr_period) / np.log10(atr_period) / np.log10((highest_high - lowest_low) + 1e-10)
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # default to 50 when range is zero
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume median, 14 for ATR
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_4h[i]) or
            np.isnan(s3_4h[i]) or
            np.isnan(r4_4h[i]) or
            np.isnan(s4_4h[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        size = fixed_size
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA34), trending market
            long_entry = (close_val > r3_4h[i]) and vol_spike and (close_val > ema_34_val) and is_trending
            # Short: price < S3 and volume spike, in downtrend (close < EMA34), trending market
            short_entry = (close_val < s3_4h[i]) and vol_spike and (close_val < ema_34_val) and is_trending
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, at R4 (take profit), or if market becomes ranging
            if close_val < ema_34_val or close_val > r4_4h[i] or not is_trending:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, at S4 (take profit), or if market becomes ranging
            if close_val > ema_34_val or close_val < s4_4h[i] or not is_trending:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0