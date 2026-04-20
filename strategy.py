#!/usr/bin/env python3
# 4h_ADX_Momentum_With_TrendFilter
# Hypothesis: Combining ADX for trend strength with momentum (ROC) and EMA trend filter captures strong trending moves while avoiding whipsaws in ranging markets.
# ADX > 25 indicates strong trend, ROC > 0 confirms bullish momentum, ROC < 0 confirms bearish momentum.
# EMA20 on 1d timeframe filters for higher timeframe trend alignment.
# Works in bull markets by capturing strong uptrends; in bear markets by capturing strong downtrends.
# Low trade frequency due to strict ADX threshold reduces fee drag.

name = "4h_ADX_Momentum_With_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA20 trend filter on daily timeframe
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate ADX (14-period) on 4h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Rate of Change (10-period) for momentum
    roc = np.full_like(close, np.nan)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(roc[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + ROC > 0 (bullish momentum) + price above daily EMA20 (uptrend) + volume
            if adx[i] > 25 and roc[i] > 0 and close[i] > ema20_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) + ROC < 0 (bearish momentum) + price below daily EMA20 (downtrend) + volume
            elif adx[i] > 25 and roc[i] < 0 and close[i] < ema20_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if ADX < 20 (trend weakening) or ROC < 0 (momentum shift) or price below daily EMA20
            if adx[i] < 20 or roc[i] < 0 or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if ADX < 20 (trend weakening) or ROC > 0 (momentum shift) or price above daily EMA20
            if adx[i] < 20 or roc[i] > 0 or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals