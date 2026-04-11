#!/usr/bin/env python3
# 6h_12h_trix_volume_regime_v1
# Strategy: 6s TRIX momentum with 12h volume confirmation and regime filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: TRIX captures momentum shifts in 6h bars. Volume confirmation filters weak moves, while a regime filter (using 12h ATR ratio) distinguishes trending vs ranging markets to avoid whipsaws. Designed for low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_trix_volume_regime_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # TRIX calculation (15-period EMA of EMA of EMA of price, then ROC)
    def ema(series, span):
        return pd.Series(series).ewm(span=span, adjust=False).values
    
    # TRIX on close
    ema1 = ema(close, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # ROC of triple EMA
    
    # 12h ATR ratio for regime filter (ATR(12) / ATR(36))
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with index
    
    atr_12 = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    atr_36 = pd.Series(tr_12h).rolling(window=36, min_periods=36).mean().values
    atr_ratio = atr_12 / atr_36  # >1 = expanding volatility (trending), <1 = contracting (ranging)
    
    # Align TRIX and ATR ratio to 6h
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # 12h volume average for confirmation (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # warmup for TRIX and ATR
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 12h 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Regime filter: ATR ratio > 1.1 = trending regime (favor momentum)
        trending_regime = atr_ratio_aligned[i] > 1.1
        
        # TRIX signals: zero-cross with momentum
        trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
        trix_cross_down = trix_aligned[i] < 0 and trix_aligned[i-1] >= 0
        
        # Entry conditions
        # Long: TRIX crosses up AND trending regime AND volume confirmation
        if trix_cross_up and trending_regime and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: TRIX crosses down AND trending regime AND volume confirmation
        elif trix_cross_down and trending_regime and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses zero in opposite direction (mean reversion in ranging markets)
        elif position == 1 and trix_cross_down:  # Exit long on TRIX cross down
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_cross_up:   # Exit short on TRIX cross up
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals