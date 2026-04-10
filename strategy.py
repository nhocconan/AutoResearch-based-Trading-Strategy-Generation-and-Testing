#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume confirmation + 1w choppiness regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channels on 1d timeframe
# - Volume filter: 1w volume > 1.3x 20-period average volume (institutional participation)
# - Regime filter: 1w Choppiness Index < 50 (trending market favors breakout continuation)
# - In trending markets (CHOP < 50): trade breakout continuation in direction of break
# - In ranging markets (CHOP >= 50): fade Donchian extremes toward midpoint (mean reversion)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "1d_donchian_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w volume spike filter
    volume_1w = df_1w['volume'].values
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1w > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # Pre-compute 1w Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    chop_raw = np.where((hh_14 - ll_14) > 0,
                        100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14),
                        50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d ATR(20) for stoploss
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_20 = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_1d[i] < donchian_mid[i] or close_1d[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_1d[i] > donchian_mid[i] or close_1d[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume filter and chop regime
            # In trending markets (CHOP < 50): breakout continuation
            # In ranging markets (CHOP >= 50): mean reversion at Donchian extremes
            if vol_spike_aligned[i]:
                if chop_aligned[i] < 50.0:  # trending market - breakout continuation
                    # Long: price breaks above Donchian high
                    if close_1d[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: price breaks below Donchian low
                    elif close_1d[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
                else:  # ranging market - mean reversion
                    # Long: price at lower Donchian band
                    if close_1d[i] <= donchian_low[i] * 1.0005:  # tiny buffer for noise
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: price at upper Donchian band
                    elif close_1d[i] >= donchian_high[i] * 0.9995:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
    
    return signals