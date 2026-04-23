#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions. In ranging markets (chop regime),
mean reversion at extremes works well. Trend filter ensures alignment with higher timeframe
direction to avoid counter-trend trades. Volume confirmation adds validity to reversals.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) to reduce churn. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate chop regime filter (14-period) on 4h data
    # Chop = log10(sum(ATR, lookback) / log10(lookback) * 100 / (max(high) - min(low)))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.log10(sum_atr_14 / 14) / np.log10(14) * 100
    chop_filter = chop > 61.8  # ranging market
    
    # Volume confirmation (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback, 14, 20)  # EMA34, Williams %R, chop, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND chop regime AND volume
            if williams_r[i] < -80 and trend_up and chop_filter[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND chop regime AND volume
            elif williams_r[i] > -20 and trend_down and chop_filter[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R >= -50 (recovered from oversold)
                if williams_r[i] >= -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R <= -50 (recovered from overbought)
                if williams_r[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA34_Trend_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0