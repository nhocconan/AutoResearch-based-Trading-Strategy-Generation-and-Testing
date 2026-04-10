#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d volume spike + 1w chop regime filter
# - Primary signal: Williams %R(14) crossing above -80 (oversold) for long or below -20 (overbought) for short
# - Volume confirmation: 1d volume > 1.3x 20-period average volume (avoid low-participation signals)
# - Regime filter: 1w Choppiness Index > 61.8 (range market) enables mean reversion at extremes
# - Works in bull/bear: In ranging markets (CHOP > 61.8), fade Williams %R extremes; in trending markets (CHOP < 38.2), momentum continuation
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "12h_1d_1w_williamsr_volume_chop_v1"
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
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
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
    
    # Pre-compute 12h Williams %R (14)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_12h) / (highest_high - lowest_low)) * -100,
                          -50)  # neutral when no range
    
    # Pre-compute 12h ATR(14) for stoploss
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R mean reversion OR stoploss hit
            if williams_r[i] > -20 or close_12h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R mean reversion OR stoploss hit
            if williams_r[i] < -80 or close_12h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume spike and chop regime filter
            # In ranging markets (CHOP > 61.8): mean reversion from extremes
            # In trending markets (CHOP < 38.2): momentum continuation
            if volume_spike_aligned[i]:
                if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                    # Long: Williams %R crosses above -80 from below (oversold bounce)
                    if williams_r[i] > -80 and williams_r[i-1] <= -80:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: Williams %R crosses below -20 from above (overbought rejection)
                    elif williams_r[i] < -20 and williams_r[i-1] >= -20:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # trending market - momentum continuation
                    # Long: Williams %R rises above -50 (bullish momentum)
                    if williams_r[i] > -50 and williams_r[i-1] <= -50:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: Williams %R falls below -50 (bearish momentum)
                    elif williams_r[i] < -50 and williams_r[i-1] >= -50:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
    
    return signals