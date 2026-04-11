#!/usr/bin/env python3
# 4h_1d_trix_volume_regime_v1
# Strategy: 4h TRIX momentum with 1d trend filter, volume confirmation, and chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX (triple-smoothed EMA) filters noise and identifies sustained momentum. Combined with 1d EMA trend filter, volume confirmation, and chop regime (avoiding range-bound markets), this strategy captures strong trending moves while avoiding whipsaws. Target: 25-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX calculation (15-period triple EMA)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # Percentage change
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close)
    valid = (max_high - min_low) > 0
    chop[valid] = 100 * np.log10(atr_sum[valid] / (max_high[valid] - min_low[valid])) / np.log10(14)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # TRIX momentum: positive = bullish momentum, negative = bearish momentum
        trix_bullish = trix[i] > 0
        trix_bearish = trix[i] < 0
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Chop regime filter: avoid range-bound markets (CHOP > 61.8)
        # Only trade when CHOP <= 61.8 (trending market)
        trending_regime = chop[i] <= 61.8
        
        # Entry conditions
        # Long: TRIX bullish AND bullish trend AND volume confirmation AND trending regime
        if trix_bullish and trend_bullish and vol_confirm and trending_regime and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: TRIX bearish AND bearish trend AND volume confirmation AND trending regime
        elif trix_bearish and trend_bearish and vol_confirm and trending_regime and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite TRIX signal OR trend reversal
        elif position == 1 and (trix[i] < 0 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix[i] > 0 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals