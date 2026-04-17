#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Strategy: 4h TRIX momentum + volume spike + Choppiness regime filter.
Long: TRIX > 0, volume > 2.0x 20-period avg, Choppiness > 61.8 (range)
Short: TRIX < 0, volume > 2.0x 20-period avg, Choppiness > 61.8 (range)
Exit: TRIX crosses zero or volume spike ends
Position size: 0.25
Designed to capture momentum bursts in ranging markets across bull/bear cycles.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA of log returns)
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[np.nan], log_returns])  # align length
    ema1 = pd.Series(log_returns).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Calculate Choppiness Index (14-period)
    atr_vals = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_vals[i] = tr
    atr_ma = pd.Series(atr_vals).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_ma.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Fix: rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 15)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Regime filter: Choppiness > 61.8 (ranging market)
        regime_filter = chop[i] > 61.8
        
        # TRIX signals
        trix_bullish = trix[i] > 0
        trix_bearish = trix[i] < 0
        
        # Trend filter: price vs 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX bullish + volume spike + ranging + price above daily EMA
            if trix_bullish and volume_filter and regime_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish + volume spike + ranging + price below daily EMA
            elif trix_bearish and volume_filter and regime_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns bearish OR volume spike ends
            if not trix_bullish or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns bullish OR volume spike ends
            if not trix_bearish or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0