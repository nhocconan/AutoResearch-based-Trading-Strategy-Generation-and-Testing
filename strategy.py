#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d HTF for entry (price touching S3/R3 levels), volume confirmation (>1.5x 20-bar avg volume), and chop regime filter (CHOP<61.8 = trending). Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear: Camarilla pivots provide adaptive support/resistance, volume confirms conviction, chop filter avoids whipsaws in ranging markets.

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
    
    # Multi-timeframe: 1d OHLC for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_df = pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d})
    camarilla_df['range'] = camarilla_df['high'] - camarilla_df['low']
    camarilla_df['S3'] = camarilla_df['close'] - 1.1 * camarilla_df['range']
    camarilla_df['R3'] = camarilla_df['close'] + 1.1 * camarilla_df['range']
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_df['S3'].values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_df['R3'].values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below S3 level
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 level
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla level touch with volume and regime confirmation
            bullish_setup = (close[i] <= s3_aligned[i]) and volume_confirmed and trending_market
            bearish_setup = (close[i] >= r3_aligned[i]) and volume_confirmed and trending_market
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals