#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h chop regime filter
# - Primary signal: Price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - Volume confirmation: 4h volume > 20-period EMA volume (avoid low-participation breakouts)
# - Regime filter: 12h Choppiness Index(14) between 38.2 and 61.8 (avoid strong trends, trade in chop)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture momentum, chop filter avoids whipsaws in strong trends,
#   volume confirmation ensures breakout validity

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Choppiness Index(14)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) over period) / log10(period)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    atr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    chop_raw = np.where(
        (chop_denominator == 0) | (atr_sum_14 == 0),
        50.0,  # neutral when no range
        100 * np.log10(atr_sum_14 / chop_denominator) / np.log10(14)
    )
    chop_12h = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    
    # Align 12h chop to 4h timeframe (completed 12h bar only)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 20-period EMA volume
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > volume_ema_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_12h_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop is between 38.2 and 61.8 (choppy/ranging market)
        in_chop_regime = (chop_12h_aligned[i] >= 38.2) and (chop_12h_aligned[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop regime ends (too trending)
            if close[i] < lowest_low_20[i] or not in_chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop regime ends
            if close[i] > highest_high_20[i] or not in_chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and chop regime
            # Long: price breaks above Donchian high AND volume regime AND in chop regime
            if (close[i] > highest_high_20[i]) and volume_regime[i] and in_chop_regime:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume regime AND in chop regime
            elif (close[i] < lowest_low_20[i]) and volume_regime[i] and in_chop_regime:
                position = -1
                signals[i] = -0.25
    
    return signals