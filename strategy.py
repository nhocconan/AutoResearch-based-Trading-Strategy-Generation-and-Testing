#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume confirmation and chop regime filter
# - Primary signal: Price touches Camarilla H3 (short) or L3 (long) levels from 1d
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Regime filter: 1d Choppiness Index > 61.8 (range-bound market for mean reversion)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Mean reversion in range markets; filters avoid trending false signals

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1 = high_1d - low_1d
    tr_2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.zeros_like(close_1d)
    mask = range_14 > 0
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    chop_filter = chop > 61.8  # Range-bound market
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # H3 = Close + 1.1*(High-Low)*1.1/4
    # L3 = Close - 1.1*(High-Low)*1.1/4
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches L3 (mean reversion target) OR stoploss hit
            if close_12h[i] <= camarilla_l3_aligned[i] or close_12h[i] < entry_price - 2.0 * atr_14_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches H3 (mean reversion target) OR stoploss hit
            if close_12h[i] >= camarilla_h3_aligned[i] or close_12h[i] > entry_price + 2.0 * atr_14_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume and chop filters
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long: price touches or crosses above L3
                if close_12h[i] >= camarilla_l3_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price touches or crosses below H3
                elif close_12h[i] <= camarilla_h3_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals