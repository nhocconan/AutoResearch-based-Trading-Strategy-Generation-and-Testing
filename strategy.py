#!/usr/bin/env python3
# 6h_trix_volume_regime_v1
# Hypothesis: 6h strategy using TRIX momentum (12-period) with volume confirmation and chop regime filter.
# TRIX filters noise and identifies momentum shifts. Volume confirms strength. Chop regime avoids trending markets.
# Works in bull/bear by capturing momentum reversals in ranging conditions (2025+ bear/range market).
# Primary timeframe: 6h, HTF: 1d for regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on primary timeframe (6h)
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, min_periods=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, min_periods=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, min_periods=12, adjust=False).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Calculate 1d Choppiness Index for regime filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14))) / log10(14)
    chop_denom = np.log10(14) * (hh_14 - ll_14)
    chop_denom = chop_denom.replace(0, 1e-10)
    chop = 100 * np.log10(atr_14 / chop_denom) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when insufficient data
    
    # Align 1d chop to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 55)
        chop_regime = chop_aligned[i] > 55
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative or volume dries up
            if trix[i] < 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive or volume dries up
            if trix[i] > 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: TRIX crosses above zero with volume
                if trix[i] > 0 and trix[i-1] <= 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: TRIX crosses below zero with volume
                elif trix[i] < 0 and trix[i-1] >= 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals