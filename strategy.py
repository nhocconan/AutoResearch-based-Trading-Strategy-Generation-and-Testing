#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume spike + 1d chop regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 4h
# - Volume confirmation: 12h volume > 1.5x 20-period average volume (avoid low-participation breakouts)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) enables mean reversion at Donchian bands
# - Works in bull/bear: In trends (CHOP < 38.2), breakouts continue; in ranges (CHOP > 61.8), fade Donchian touches
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(20)

name = "4h_12h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h volume spike filter
    volume_12h = df_12h['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute 1d Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    chop_raw = np.where((hh_14 - ll_14) > 0,
                        100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14),
                        50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw, additional_delay_bars=0)
    
    # Pre-compute 4h Donchian Channel (20)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] < donchian_mid[i] or close_4h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] > donchian_mid[i] or close_4h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and chop regime filter
            # In ranging markets (CHOP > 61.8): fade Donchian touches (mean reversion)
            # In trending markets (CHOP < 38.2): breakout continuation
            if volume_spike_aligned[i]:
                if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                    # Long: price touches lower Donchian band
                    if close_4h[i] <= donchian_low[i] * 1.001:  # small buffer for noise
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price touches upper Donchian band
                    elif close_4h[i] >= donchian_high[i] * 0.999:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # trending market - breakout continuation
                    # Long: price breaks above upper Donchian band
                    if close_4h[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price breaks below lower Donchian band
                    elif close_4h[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
    
    return signals