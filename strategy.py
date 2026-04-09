#!/usr/bin/env python3
# 4h_hma_trend_volume_chop_v1
# Hypothesis: 4h strategy using HMA trend filter from 1d timeframe, volume confirmation, and chop regime filter.
# In both bull and bear markets, price tends to follow the higher timeframe trend when volume confirms and market is not too choppy.
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Target: 75-200 total trades over 4 years.
# Primary timeframe: 4h, HTF: 1d for HMA and chop filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_hma_trend_volume_chop_v1"
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
    
    # 1d HTF data for HMA and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate HMA (Hull Moving Average) on 1d close - 21 period
    def hull_moving_average(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(arr, window):
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights, 'valid') / weights.sum()
        
        wma_half = wma(data, half_period)
        wma_full = wma(data, period)
        hull_raw = 2 * wma_half - wma_full
        hma = wma(hull_raw, sqrt_period)
        
        # Pad with NaN to match original length
        hma_padded = np.full_like(data, np.nan)
        hma_padded[period-1:period-1+len(hma)] = hma
        return hma_padded
    
    # Calculate HMA on 1d close
    hma_1d = hull_moving_average(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close[0])
    tr3[0] = np.abs(low_1d[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average for confirmation (20-period) on 4h
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50) or trending (chop < 50) - adaptive
        # In choppy markets (>50), we mean revert; in trending markets (<50), we follow trend
        chop_value = chop_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR volume dries up OR extreme chop
            if hma_1d_aligned[i] < close[i] or not volume_confirmed or chop_value > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR volume dries up OR extreme chop
            if hma_1d_aligned[i] > close[i] or not volume_confirmed or chop_value > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed:
                # Adaptive entry based on chop regime
                if chop_value > 50:  # Choppy market - mean reversion
                    # Long when price is below HMA (oversold in chop)
                    if close[i] < hma_1d_aligned[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short when price is above HMA (overbought in chop)
                    elif close[i] > hma_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.30
                else:  # Trending market - follow trend
                    # Long when price is above HMA (uptrend)
                    if close[i] > hma_1d_aligned[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short when price is below HMA (downtrend)
                    elif close[i] < hma_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.30
    
    return signals