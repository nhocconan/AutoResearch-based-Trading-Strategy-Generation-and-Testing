#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_regime_v1
# Hypothesis: 4h strategy using Camarilla pivot levels from 1d HTF for entry (price touches L3/H3 or L4/H4 levels), volume confirmation (>1.5x 20-bar avg volume), and chop regime filter (CHOP<61.8 = trending). Discrete position sizing (0.25) to minimize fee churn. Target: 19-50 trades/year (75-200 total over 4 years). Camarilla pivots provide statistical support/resistance levels that work in both trending and ranging markets; volume confirms institutional participation; chop filter avoids whipsaws in low-volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_regime_v1"
timeframe = "4h"
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
    
    # Multi-timeframe: 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_high = np.zeros_like(close_1d)
    camarilla_low = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            camarilla_high[i] = np.nan
            camarilla_low[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            continue
            
        range_1d = high_1d[i] - low_1d[i]
        camarilla_high[i] = close_1d[i] + range_1d * 1.1 / 12  # H4
        camarilla_low[i] = close_1d[i] - range_1d * 1.1 / 12   # L4
        camarilla_h3[i] = close_1d[i] + range_1d * 1.1 / 6    # H3
        camarilla_l3[i] = close_1d[i] - range_1d * 1.1 / 6    # L3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
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
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below L3 level
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 level
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla level touch with volume and regime confirmation
            # Long when price touches or crosses above L3/L4 with bullish bias
            bullish_setup = ((close[i] <= camarilla_l3_aligned[i]) or (close[i] <= camarilla_l4_aligned[i])) and volume_confirmed and trending_market
            # Short when price touches or crosses below H3/H4 with bearish bias
            bearish_setup = ((close[i] >= camarilla_h3_aligned[i]) or (close[i] >= camarilla_h4_aligned[i])) and volume_confirmed and trending_market
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals