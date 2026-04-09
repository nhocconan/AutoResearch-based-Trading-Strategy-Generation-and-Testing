#!/usr/bin/env python3
# 1d_donchian_volume_chop_regime_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation (>1.5x 20-day average) and weekly chop regime filter (CHOP(14) > 61.8 = range, < 38.2 = trend).
# In ranging markets (CHOP > 61.8): mean reversion at Donchian bands (short at upper band, long at lower band).
# In trending markets (CHOP < 38.2): trend continuation (breakout long at upper band, short at lower band).
# Uses 1d primary timeframe, 1w HTF for chop regime. Discrete sizing (±0.25) minimizes fee churn.
# Target: 15-25 trades/year to avoid fee drag in bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian(20) - upper/lower bands
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w HTF data for chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for weekly ATR(14) and HH/LL(14)
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly ATR(14) for chop denominator
    tr_w = np.maximum(
        np.maximum(
            np.abs(high_w[1:] - low_w[1:]),
            np.abs(high_w[1:] - close_w[:-1])
        ),
        np.abs(low_w[1:] - close_w[:-1])
    )
    tr_w = np.concatenate([[np.nan], tr_w])  # align length
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly highest high and lowest low over 14 periods
    highest_high_14w = pd.Series(high_w).rolling(window=14, min_periods=14).max().values
    lowest_low_14w = pd.Series(low_w).rolling(window=14, min_periods=14).min().values
    
    # Chopiness index formula: 100 * log10(sum(ATR(14)) / (max(HH,14) - min(LL,14))) / log10(14)
    sum_atr_14w = pd.Series(atr_14_w).rolling(window=14, min_periods=14).sum().values
    range_14w = highest_high_14w - lowest_low_14w
    chop_denom = np.where(range_14w > 0, range_14w, np.nan)
    chop_raw = 100 * np.log10(sum_atr_14w / chop_denom) / np.log10(14)
    chop_w = chop_raw  # Already in [0, 100] range
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        chop = chop_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop > 61.8:  # ranging market: mean reversion exit at opposite band
                if close[i] >= lowest_low[i]:  # price retraced to lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # trending market: trail with Donchian lower band
                if close[i] <= lowest_low[i]:  # broke below lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop > 61.8:  # ranging market: mean reversion exit at opposite band
                if close[i] <= highest_high[i]:  # price retraced to upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # trending market: trail with Donchian upper band
                if close[i] >= highest_high[i]:  # broke above upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if chop > 61.8:  # ranging market: mean reversion at bands
                if close[i] <= lowest_low[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= highest_high[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # trending market: breakout continuation
                if close[i] >= highest_high[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] <= lowest_low[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals