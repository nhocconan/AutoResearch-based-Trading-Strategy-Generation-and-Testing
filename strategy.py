#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter on 12h timeframe.
Only trade in direction of 1d trend when market is trending (CHOP < 61.8) and volume > 1.5x 20-period median.
Uses discrete sizing 0.25 to target 12-37 trades/year. Works in bull/bear via 1d trend filter and avoids choppy markets via regime filter.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
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
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Choppiness regime filter: CHOP(14) < 61.8 = trending regime (good for breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log10(hh14 - ll14) / np.log10(14)) if (hh14 - ll14) > 0 else 100
    # Vectorized chop calculation
    atr14_series = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    hh14_series = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll14_series = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop_series = 100 * np.log10(atr14_series) / np.log10(hh14_series - ll14_series) / np.log10(14)
    chop_series = chop_series.fillna(50).replace([np.inf, -np.inf], 50).values
    chop_filter = chop_series < 61.8  # trending regime
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume median, 14 for chop
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or
            np.isnan(r4_12h[i]) or
            np.isnan(s4_12h[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop_series[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        regime_ok = chop_series[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike and trending regime, in uptrend (close > EMA34)
            long_entry = (close_val > r3_12h[i]) and vol_spike and regime_ok and (close_val > ema_34_val)
            # Short: price < S3 and volume spike and trending regime, in downtrend (close < EMA34)
            short_entry = (close_val < s3_12h[i]) and vol_spike and regime_ok and (close_val < ema_34_val)
            
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
            # Long - exit on trend reversal or at R4 (take profit) or chop regime becomes too high
            if close_val < ema_34_val or close_val > r4_12h[i] or chop_series[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit) or chop regime becomes too high
            if close_val > ema_34_val or close_val < s4_12h[i] or chop_series[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0