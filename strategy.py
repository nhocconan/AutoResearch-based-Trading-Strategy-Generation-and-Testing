#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d HTF for mean-reversion entries, volume confirmation (>1.5x 20-bar avg volume), and chop regime filter (CHOP>61.8 = ranging). Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear: Camarilla pivots act as support/resistance in ranging markets, volume confirms conviction at pivot touches, chop filter ensures ranging regime where mean reversion works.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    # Avoid division by zero or log of zero
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h, 2 bars per 1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches camarilla H3 (take profit) or H4 (stop)
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches camarilla L3 (take profit) or L4 (stop)
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for mean reversion at camarilla levels with volume and regime confirmation
            # Long when price touches/slightly breaks L4 and reverses up
            long_setup = (close[i] <= camarilla_l4_aligned[i] * 1.001) and volume_confirmed and ranging_market
            # Short when price touches/slightly breaks H4 and reverses down
            short_setup = (close[i] >= camarilla_h4_aligned[i] * 0.999) and volume_confirmed and ranging_market
            
            if long_setup:
                position = 1
                signals[i] = 0.25
            elif short_setup:
                position = -1
                signals[i] = -0.25
    
    return signals