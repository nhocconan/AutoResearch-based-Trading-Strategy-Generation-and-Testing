#!/usr/bin/env python3
# 4h_trix_volume_chop_v1
# Hypothesis: 4h strategy using TRIX momentum with volume confirmation and chop regime filter.
# TRIX (12) filters noise and identifies sustainable momentum. Volume confirmation ensures
# institutional participation. Chop filter (from 1d) ensures we trade in trending regimes
# (chop < 50) to avoid whipsaws in ranging markets. Works in bull/bear by capturing
# sustained moves with proper regime filtering. Target: 25-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_chop_v1"
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
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (12) - triple smoothed EMA of ROC
    close_s = pd.Series(close)
    # ROC(1): (close[t] - close[t-1]) / close[t-1]
    roc = close_s.pct_change(1)
    # EMA1 of ROC
    ema1 = roc.ewm(span=12, min_periods=12, adjust=False).mean()
    # EMA2 of EMA1
    ema2 = ema1.ewm(span=12, min_periods=12, adjust=False).mean()
    # EMA3 of EMA2 (TRIX)
    ema3 = ema2.ewm(span=12, min_periods=12, adjust=False).mean()
    trix = ema3.values * 100  # Scale for readability
    
    # Calculate chop regime (14-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is trending (chop < 50)
        chop_regime = chop_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative or volume dries up
            if trix[i] < 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive or volume dries up
            if trix[i] > 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: TRIX positive with volume confirmation
                if trix[i] > 0:
                    position = 1
                    signals[i] = 0.30
                # Short entry: TRIX negative with volume confirmation
                elif trix[i] < 0:
                    position = -1
                    signals[i] = -0.30
    
    return signals