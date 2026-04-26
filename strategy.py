#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA50 trend filter, volume confirmation, and choppiness regime filter. 
Designed for low trade frequency (<50/year) to minimize fee drag. Uses discrete sizing 0.25. 
Choppiness filter avoids whipsaws in ranging markets. Works in both bull/bear via trend filter.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r3 = close_1d + 1.125 * rng
    s3 = close_1d - 1.125 * rng
    r4 = close_1d + 1.5 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 1.8x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.8 * vol_median_20)
    
    # Choppiness regime filter on 1d: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # Calculate CHOP(14) on 1d data
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high - lowest_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop = 100 * np.log10(sum_tr / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    # Only trade when market is trending (CHOP < 38.2)
    trending_regime = chop_aligned < 38.2
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for EMA, 14 for ATR/CHOP, 20 for volume median
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(r3_4h[i]) or
            np.isnan(s3_4h[i]) or
            np.isnan(r4_4h[i]) or
            np.isnan(s4_4h[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_trending = trending_regime[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry ONLY in trending regime
            if not is_trending:
                signals[i] = 0.0
                continue
            # Long: price > R3 and volume spike, in uptrend (close > EMA50_1d)
            long_entry = (close_val > r3_4h[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA50_1d)
            short_entry = (close_val < s3_4h[i]) and vol_spike and (close_val < ema_50_val)
            
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
            # Long - exit on trend reversal, ATR stoploss, or at R4 (take profit)
            stop_price = entry_price - 2.5 * atr_val
            if (close_val < ema_50_val) or (close_val < stop_price) or (close_val > r4_4h[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at S4 (take profit)
            stop_price = entry_price + 2.5 * atr_val
            if (close_val > ema_50_val) or (close_val > stop_price) or (close_val < s4_4h[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0