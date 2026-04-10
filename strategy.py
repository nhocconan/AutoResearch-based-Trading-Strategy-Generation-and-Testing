#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels + 1d volume spike + 1w ADX regime filter
# - Primary signal: Price reverses from Camarilla H3/L3 levels (mean reversion in ranges, continuation in trends)
# - Volume confirmation: 1d volume > 1.3x 20-period average (avoid low-participation false signals)
# - Regime filter: 1w ADX > 25 indicates trending market (breakout continuation), ADX < 20 indicates ranging (mean reversion)
# - Works in bull/bear: In trending markets (ADX > 25), trade Camarilla breakouts; in ranging markets (ADX < 20), fade H3/L3 touches
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_1w_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    close_1d = df_1w['close'].values if len(df_1w) > 0 else np.array([])  # placeholder, will use 1d data below
    # Actually use 1d data for Camarilla calculation
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla levels: based on previous day's range
        # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
        # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
        range_1d = high_1d - low_1d
        camarilla_h3 = close_1d + 1.125 * range_1d
        camarilla_l3 = close_1d - 1.125 * range_1d
        camarilla_h4 = close_1d + 1.5 * range_1d
        camarilla_l4 = close_1d - 1.5 * range_1d
        
        # Align to 4h timeframe
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        h3_aligned = l3_aligned = h4_aligned = l4_aligned = np.full(n, np.nan)
    
    # Pre-compute 4h ATR(20) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
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
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: mean reversion at midpoint OR stoploss hit
            mid_level = (h3_aligned[i] + l3_aligned[i]) / 2
            if close_4h[i] < mid_level or close_4h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: mean reversion at midpoint OR stoploss hit
            mid_level = (h3_aligned[i] + l3_aligned[i]) / 2
            if close_4h[i] > mid_level or close_4h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume spike and ADX regime filter
            # In ranging markets (ADX < 20): fade H3/L3 touches (mean reversion)
            # In trending markets (ADX > 25): breakout continuation from H3/L3
            if volume_spike_aligned[i]:
                if adx_aligned[i] < 20:  # ranging market - mean reversion
                    # Long: price touches L3 level
                    if close_4h[i] <= l3_aligned[i] * 1.002:  # small buffer for noise
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price touches H3 level
                    elif close_4h[i] >= h3_aligned[i] * 0.998:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
                elif adx_aligned[i] > 25:  # trending market - breakout continuation
                    # Long: price breaks above H3 level
                    if close_4h[i] > h3_aligned[i] * 1.002:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price breaks below L3 level
                    elif close_4h[i] < l3_aligned[i] * 0.998:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
    
    return signals