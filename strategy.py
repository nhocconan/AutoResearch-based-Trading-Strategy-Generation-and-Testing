#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w Choppiness Index regime filter
# - Primary signal: Williams %R(14) oversold/overbought on daily timeframe
# - Regime filter: 1w Choppiness Index > 61.8 (range market) enables mean reversion at extremes
# - In ranging markets (CHOP > 61.8): fade Williams %R extremes (mean reversion)
# - In trending markets (CHOP < 38.2): follow Williams %R momentum (continuation)
# - Volume confirmation: 1d volume > 1.3x 20-period average volume
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 30-100 trades over 4 years (7-25/year) per 1d strategy guidelines

name = "1d_williamsr_chop_volume_v1"
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
    
    # Pre-compute 1d Williams %R(14)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) > 0,
                          ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100,
                          -50)  # neutral when no range
    
    # Pre-compute 1d volume spike filter
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.3 * avg_volume_20)
    
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R mean reversion (return to -50) OR adverse move
            if williams_r[i] > -50:  # returned to midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R mean reversion (return to -50) OR adverse move
            if williams_r[i] < -50:  # returned to midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume spike and chop regime filter
            # In ranging markets (CHOP > 61.8): mean reversion at extremes
            # In trending markets (CHOP < 38.2): momentum continuation
            if volume_spike[i]:
                if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                    # Long: Williams %R oversold (< -80)
                    if williams_r[i] < -80:
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: Williams %R overbought (> -20)
                    elif williams_r[i] > -20:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # trending market - momentum continuation
                    # Long: Williams %R rising from oversold (> -80 and rising)
                    if williams_r[i] > -80 and i > 100 and williams_r[i] > williams_r[i-1]:
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: Williams %R falling from overbought (< -20 and falling)
                    elif williams_r[i] < -20 and i > 100 and williams_r[i] < williams_r[i-1]:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
    
    return signals