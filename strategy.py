#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
    # Long: price breaks above Donchian(20) high AND 1d volume > 1.5 * 20-period MA AND chop > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND 1d volume > 1.5 * 20-period MA AND chop > 61.8 (range)
    # Exit: price returns to Donchian(20) midpoint OR chop < 38.2 (trend)
    # Uses 12h for price action/Donchian, 1d for volume/chop filters
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Donchian channels (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donch_mid_12h = (donch_high_12h + donch_low_12h) / 2.0
    
    # Align 12h Donchian to 12h timeframe (no additional delay for price-based indicators)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid_12h)
    
    # Calculate 1d volume spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - sum of TR over 14 periods
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero when HH14 == LL14
    range_14 = hh_14 - ll_14
    chop_1d = np.full_like(atr_1d, np.nan)
    mask = (range_14 > 0) & (~np.isnan(atr_1d)) & (~np.isnan(range_14))
    chop_1d[mask] = 100 * np.log10(atr_1d[mask] / range_14[mask]) / np.log10(14)
    
    # Chop regime filters: > 61.8 = range (favor mean reversion/breakouts), < 38.2 = trending
    chop_range = chop_1d > 61.8
    chop_trend = chop_1d < 38.2
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime: only trade in choppy markets (range-bound)
        in_chop = chop_range_aligned[i] > 0.5
        # Exit chop regime: trend developing
        exiting_chop = chop_trend_aligned[i] > 0.5
        
        # Donchian breakout signals
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        return_to_mid = abs(close[i] - donch_mid_aligned[i]) < 0.001 * close[i]  # within 0.1% of midpoint
        
        # Volume confirmation
        vol_confirmed = volume_spike_aligned[i] > 0.5
        
        # Entry logic: Donchian breakout + volume spike + chop regime
        long_entry = breakout_up and vol_confirmed and in_chop
        short_entry = breakout_down and vol_confirmed and in_chop
        
        # Exit logic: return to midpoint OR chop regime ends (trend developing)
        long_exit = return_to_mid or exiting_chop
        short_exit = return_to_mid or exiting_chop
        
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

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0