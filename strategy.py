#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long when price breaks above upper Donchian(20) + volume > 1.5x 20-period average + chop < 61.8 (trending).
# Short when price breaks below lower Donchian(20) + volume > 1.5x 20-period average + chop < 61.8 (trending).
# Exit when price closes back inside Donchian(20) channel.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong trending moves while avoiding range-bound false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=20, min_periods=20).max().values
    lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Chop regime filter from 1d timeframe (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) for chop denominator
    atr_s = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sumTR(14) / (ATR(14) * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_s * 14)) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: trending market (chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back below lower Donchian (mean reversion) or above upper (trailing)
            if close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above upper Donchian (mean reversion) or below lower (trailing)
            if close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and trending regime
            bullish_breakout = (close[i] > upper[i]) and volume_confirmed and trending_regime
            bearish_breakout = (close[i] < lower[i]) and volume_confirmed and trending_regime
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals