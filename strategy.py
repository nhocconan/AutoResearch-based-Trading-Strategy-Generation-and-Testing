#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with 1d Donchian breakout and volume confirmation.
# Uses daily Choppiness Index > 61.8 for range conditions, then mean-reverts at Donchian bands.
# In trending markets (CHOP < 38.2), follows Donchian breakouts.
# Volume spike confirms signal strength. Designed for low-turnover, regime-adaptive trading.
# Target: 12-30 trades/year to stay within optimal range for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness, Donchian, and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Choppiness Index on daily data
    # CHOP = 100 * log10(SUM(ATR1) / (MAX(HIGH) - MIN(LOW))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align indices
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr1 / range_14) / np.log10(14)
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Volume filter: current volume > 2.5 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian + 14-period Chop + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_12h[i]) or 
            np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.5x average (strict)
        volume_filter = volume[i] > (2.5 * volume_ma20[i])
        
        # Regime filters
        chop_high = chop_12h[i] > 61.8  # ranging market
        chop_low = chop_12h[i] < 38.2   # trending market
        
        # Price relative to 1d Donchian channels
        price_above_high = close[i] > donchian_high_12h[i]
        price_below_low = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # In ranging market: mean reversion at Donchian bands
            if chop_high:
                # Long at lower band with volume
                if price_below_low and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short at upper band with volume
                elif price_above_high and volume_filter:
                    signals[i] = -0.25
                    position = -1
            # In trending market: follow breakout
            elif chop_low:
                # Long breakout above upper band
                if price_above_high and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short breakout below lower band
                elif price_below_low and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if chop_high:
                # In range: exit at opposite band
                if close[i] > donchian_high_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trend: exit on reversal or volume drop
                if close[i] < donchian_low_12h[i] or not volume_filter:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if chop_high:
                # In range: exit at opposite band
                if close[i] < donchian_low_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trend: exit on reversal or volume drop
                if close[i] > donchian_high_12h[i] or not volume_filter:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Chop_Donchian_Volume_Regime"
timeframe = "12h"
leverage = 1.0