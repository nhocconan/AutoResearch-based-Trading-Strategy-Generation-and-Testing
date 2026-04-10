#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR volume filter + 1w Choppiness regime
# - Primary signal: Price breaks above/below 20-period Donchian channel on 12h with volume confirmation
# - Volatility filter: 1d ATR(14) > 1.2x 20-period average ATR (ensures sufficient volatility for breakout)
# - Regime filter: 1w Choppiness Index > 50 (ranging market = mean reversion at Donchian extremes)
# - In ranging markets (CHOP > 50): fade Donchian extremes toward mid-line (mean reversion)
# - In trending markets (CHOP <= 50): breakout continuation in direction of break
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "12h_1d_1w_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d ATR volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    avg_atr_20 = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_14_1d > (1.2 * avg_atr_20)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
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
    
    # Pre-compute 12h Donchian Channel (20)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 12h ATR(20) for stoploss
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_20 = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_filter_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_12h[i] < donchian_mid[i] or close_12h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_12h[i] > donchian_mid[i] or close_12h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volatility filter and chop regime
            # In ranging markets (CHOP > 50): mean reversion at Donchian extremes
            # In trending markets (CHOP <= 50): breakout continuation
            if vol_filter_aligned[i]:
                if chop_aligned[i] > 50.0:  # ranging market - mean reversion
                    # Long: price at lower Donchian band
                    if close_12h[i] <= donchian_low[i] * 1.0005:  # tiny buffer for noise
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: price at upper Donchian band
                    elif close_12h[i] >= donchian_high[i] * 0.9995:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
                else:  # trending market - breakout continuation
                    # Long: price breaks above upper Donchian band
                    if close_12h[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: price breaks below lower Donchian band
                    elif close_12h[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
    
    return signals