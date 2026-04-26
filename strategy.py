#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter, volume confirmation, and choppiness regime filter. Designed to work in both bull and bear markets by using the 12h trend to filter direction and choppiness index to avoid whipsaws in ranging markets. Uses discrete sizing 0.25 to minimize fee churn and targets 20-50 trades/year via tight entry conditions (trend + volume + breakout + regime). Includes ATR-based stoploss for risk control.
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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
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
    r1 = close_1d + 1.125 * rng / 12
    s1 = close_1d - 1.125 * rng / 12
    r2 = close_1d + 1.125 * rng / 6
    s2 = close_1d - 1.125 * rng / 6
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 4h choppiness index regime filter
    chop_window = 14
    chop = np.full(n, np.nan)
    for i in range(chop_window, n):
        high_slice = high[i-chop_window+1:i+1]
        low_slice = low[i-chop_window+1:i+1]
        atr_sum = np.sum(np.maximum(np.abs(high_slice - low_slice), 
                                   np.maximum(np.abs(high_slice - np.roll(close[i-chop_window+1:i+1], 1)),
                                            np.abs(low_slice - np.roll(close[i-chop_window+1:i+1], 1)))))
        if atr_sum > 0:
            chop[i] = 100 * np.log10(chop_window / np.log2(chop_window)) / np.log10(atr_sum)
        else:
            chop[i] = 50.0  # neutral when no movement
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 12h EMA, 14 for ATR, 20 for volume median, 14 for chop
    start_idx = max(50, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(r1_4h[i]) or
            np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or
            np.isnan(s2_4h[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R1 and volume spike, in uptrend (close > EMA50_12h), chop < 61.8 (trending)
            long_entry = (close_val > r1_4h[i]) and vol_spike and (close_val > ema_50_val) and (chop_val < 61.8)
            # Short: price < S1 and volume spike, in downtrend (close < EMA50_12h), chop < 61.8 (trending)
            short_entry = (close_val < s1_4h[i]) and vol_spike and (close_val < ema_50_val) and (chop_val < 61.8)
            
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
            if close_val < ema_50_val or close_val < stop_price or close_val > r2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at S2 (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_50_val or close_val > stop_price or close_val < s2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0