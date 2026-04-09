#!/usr/bin/env python3
# 6h_trix_volume_regime_v1
# Hypothesis: 6h TRIX (triple EMA) crossover with volume confirmation and 1d chop regime filter.
# Works in bull/bear: TRIX captures momentum shifts; volume confirms institutional participation;
# chop regime avoids whipsaws in ranging markets. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for chop calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Chopiness Index (CHOP) - measures whether market is choppy (trendless) or trending
    # CHOP > 61.8 = ranging/choppy market (mean revert)
    # CHOP < 38.2 = strongly trending market (trend follow)
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.maximum(high_1d - low_1d,
                       np.absolute(high_1d - np.roll(close_1d, 1)),
                       np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high_1d - lowest_low_1d
    denominator = np.where(denominator == 0, 1, denominator)
    
    chop_1d = 100 * np.log10(np.sum(atr_1d) / denominator * np.sqrt(14)) / np.log10(np.sqrt(14))
    # Fix: Calculate properly for each bar
    chop_1d = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(tr_1d[i-13:i+1])  # 14-period ATR sum
        hh = highest_high_1d[i]
        ll = lowest_low_1d[i]
        if hh - ll > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / (hh - ll) * np.sqrt(14)) / np.log10(np.sqrt(14))
        else:
            chop_1d[i] = 50  # Neutral when range is zero
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 6h TRIX (Triple Exponential Average) - momentum oscillator
    # TRIX = % change in triple-smoothed EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.zeros(n)
    trix[12:] = (ema3[12:] - ema3[11:-1]) / ema3[11:-1] * 100  # Percentage change
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative OR chop regime too high (choppy market)
            if trix[i] < 0 or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive OR chop regime too high (choppy market)
            if trix[i] > 0 or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trending regime (CHOP < 38.2)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            trending_regime = chop_1d_aligned[i] < 38.2
            
            if volume_confirmed and trending_regime:
                # Long: TRIX crosses above zero with rising momentum
                if trix[i] > 0 and trix[i] > trix[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: TRIX crosses below zero with falling momentum
                elif trix[i] < 0 and trix[i] < trix[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals