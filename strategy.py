#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 12h choppiness regime filter
    # Long: price breaks above Donchian(20) high AND 1d volume > 1.5 * 20-period average AND 12h CHOP > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND 1d volume > 1.5 * 20-period average AND 12h CHOP > 61.8 (range)
    # Exit: price reverts to Donchian(20) midpoint OR 12h CHOP < 38.2 (trend)
    # Uses 4h for Donchian breakouts, 1d for volume confirmation, 12h for chop regime
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year) within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for choppiness regime (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (no additional delay for price-based indicators)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Calculate 1d volume spike confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    
    # Align 1d volume spike to 1d timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 12h choppiness regime: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with index 0
    
    # ATR(14) for 12h
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods for 12h
    hh_14_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero and handle NaN
    hh_ll_diff_12h = hh_14_12h - ll_14_12h
    sum_atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    chop_12h = np.full_like(atr_12h, np.nan)
    valid_chop = (hh_ll_diff_12h > 0) & ~np.isnan(sum_atr_12h) & ~np.isnan(hh_ll_diff_12h)
    chop_12h[valid_chop] = 100 * np.log10(sum_atr_12h[valid_chop] / hh_ll_diff_12h[valid_chop]) / np.log10(14)
    
    # Align 12h chop to 12h timeframe (wait for completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h CHOP > 61.8 (range-bound market)
        range_regime = chop_aligned[i] > 61.8
        # Exit regime: CHOP < 38.2 (trending market)
        trend_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = volume_spike_aligned[i] > 0.5  # treated as boolean
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation + range regime
        long_entry = long_breakout and vol_confirmed and range_regime
        short_entry = short_breakout and vol_confirmed and range_regime
        
        # Exit logic: price reverts to midpoint OR regime shifts to trend
        long_exit = (close[i] <= donchian_mid_aligned[i]) or trend_regime
        short_exit = (close[i] >= donchian_mid_aligned[i]) or trend_regime
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0