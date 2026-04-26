#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout on 4h with 1-day EMA34 trend filter, volume spike, and choppiness regime filter. Designed to work in both bull and bear markets by using 1-day trend for direction and chop filter to avoid false breakouts in ranging markets. Uses discrete sizing 0.30 to balance return and risk, targeting ~30-50 trades/year via tight entry conditions (trend + volume + breakout + chop). Includes ATR-based stoploss for risk control.
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
    
    # Calculate Camarilla R1, S1, R2, S2 for each 1d bar
    rng = high_1d - low_1d
    r1 = close_1d + 1.125 * rng * (1/6)  # Camarilla R1
    s1 = close_1d - 1.125 * rng * (1/6)  # Camarilla S1
    r2 = close_1d + 1.125 * rng * (2/6)  # Camarilla R2
    s2 = close_1d - 1.125 * rng * (2/6)  # Camarilla S2
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = np.where((highest_high - lowest_low) > 0,
                        100 * np.log10(np.sum(atr_14)) / np.log10(14) / np.log10(highest_high - lowest_low),
                        50)
    # Fix: proper CHOP calculation
    sum_atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = np.where((highest_high - lowest_low) > 0,
                    100 * np.log10(sum_atr_14 / (highest_high - lowest_low)) / np.log10(14),
                    50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 14 for ATR/CHOP, 20 for volume median
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(r1_4h[i]) or
            np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or
            np.isnan(s2_4h[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Regime filter: CHOP > 50 = ranging (mean revert), CHOP < 50 = trending (trend follow)
        # We use CHOP < 50 for trend following breakouts
        in_trending_regime = chop_val < 50
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R1 and volume spike, in uptrend (close > EMA34_1d), trending regime
            long_entry = (close_val > r1_4h[i]) and vol_spike and (close_val > ema_34_val) and in_trending_regime
            # Short: price < S1 and volume spike, in downtrend (close < EMA34_1d), trending regime
            short_entry = (close_val < s1_4h[i]) and vol_spike and (close_val < ema_34_val) and in_trending_regime
            
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
            # Long - exit on trend reversal, ATR stoploss, or at R2 (take profit)
            stop_price = entry_price - 2.0 * atr_val
            if close_val < ema_34_val or close_val < stop_price or close_val > r2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at S2 (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_34_val or close_val > stop_price or close_val < s2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0