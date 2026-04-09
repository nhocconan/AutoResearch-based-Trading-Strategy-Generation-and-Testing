#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_reversal_volume_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels (H3/L3) for mean reversion entries.
# Long: Price touches or crosses below daily L3, volume > 1.3x 20-period average, and chop regime (CHOP > 61.8).
# Short: Price touches or crosses above daily H3, volume > 1.3x 20-period average, and chop regime (CHOP > 61.8).
# Exit: Price reverts to the 12h 20-period EMA or opposite pivot level (H3 for long, L3 for short).
# Uses daily Camarilla H3/L3 for mean reversion structure, volume to confirm interest, chop filter to avoid strong trends.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_reversal_volume_v1"
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
    
    # ATR(14) for choppiness index calculation
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR14) / (max(high, lookback) - min(low, lookback))) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = high_s.rolling(window=14, min_periods=14).max().values
    ll = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 1d OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h 20-period EMA for exit
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(ema_20[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price reverts to EMA20 or breaks above H3 (invalidates mean reversion)
            if close[i] >= ema_20[i] or high[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reverts to EMA20 or breaks below L3 (invalidates mean reversion)
            if close[i] <= ema_20[i] or low[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches/crosses below L3, volume confirmed, and chop regime
            if (low[i] <= camarilla_l3_aligned[i] and volume_confirmed and chop_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches/crosses above H3, volume confirmed, and chop regime
            elif (high[i] >= camarilla_h3_aligned[i] and volume_confirmed and chop_filter):
                position = -1
                signals[i] = -0.25
    
    return signals