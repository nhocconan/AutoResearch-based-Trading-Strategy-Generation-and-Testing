#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
    # Long: price breaks above Donchian upper band AND 1d volume > 1.5 * 20-period MA AND chop > 61.8 (range)
    # Short: price breaks below Donchian lower band AND 1d volume > 1.5 * 20-period MA AND chop > 61.8 (range)
    # Exit: price crosses Donchian midline OR chop < 38.2 (trend)
    # Uses 12h for Donchian breakout (structure), 1d for volume/chop (regime/context)
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20): upper = max(high,20), lower = min(low,20), mid = (upper+lower)/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper band (20-period high)
    donchian_upper_12h = np.full_like(high_12h, np.nan)
    for i in range(19, len(high_12h)):
        donchian_upper_12h[i] = np.max(high_12h[i-19:i+1])
    
    # Donchian lower band (20-period low)
    donchian_lower_12h = np.full_like(low_12h, np.nan)
    for i in range(19, len(low_12h)):
        donchian_lower_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Donchian midline
    donchian_mid_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # Align 12h Donchian levels to 12h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    
    # Calculate 1d volume spike: volume > 1.5 * 20-period MA
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    volume_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) - measures trend vs range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - smoothed TR
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_14_1d = np.full_like(high_1d, np.nan)
    ll_14_1d = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        hh_14_1d[i] = np.max(high_1d[i-13:i+1])
        ll_14_1d[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR(14)) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = np.full_like(atr_14_1d, np.nan)
    for i in range(13, len(atr_14_1d)):
        sum_tr_14[i] = np.sum(atr_14_1d[i-13:i+1])
    
    chop_1d = np.full_like(close_1d, np.nan)
    mask = (~np.isnan(sum_tr_14) & ~np.isnan(hh_14_1d) & ~np.isnan(ll_14_1d) & 
            ((hh_14_1d - ll_14_1d) > 0))
    chop_1d[mask] = 100 * np.log10(sum_tr_14[mask] / (hh_14_1d[mask] - ll_14_1d[mask])) / np.log10(14)
    
    # Align 1d volume spike and chop to 12h (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop > 61.8 (range-bound market)
        range_regime = chop_aligned[i] > 61.8
        # Exit regime: chop < 38.2 (trending market)
        trend_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = bool(volume_spike_aligned[i])
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Entry logic: Donchian breakout + volume spike + range regime
        long_entry = long_breakout and vol_confirmed and range_regime
        short_entry = short_breakout and vol_confirmed and range_regime
        
        # Exit logic: price crosses Donchian midline OR regime shifts to trend
        long_exit = (close[i] < donchian_mid_aligned[i]) or trend_regime
        short_exit = (close[i] > donchian_mid_aligned[i]) or trend_regime
        
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

name = "12h_1d_donchian_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0