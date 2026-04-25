#!/usr/bin/env python3
"""
6h_AdaptiveVolatilityBreakout_RegimeFilter_v1
Hypothesis: Trade 6h volatility breakouts with adaptive thresholds based on ATR percentile and regime filter (ADX). 
Long when price breaks above upper band (close + k*ATR) in trending up regime (ADX>25 + +DI>-DI). 
Short when price breaks below lower band (close - k*ATR) in trending down regime (ADX>25 + +DI<+DI). 
k adapts to ATR percentile: tighter in low vol (k=1.0), wider in high vol (k=2.0). 
Volume confirmation required. Position size 0.25. Target 50-150 trades over 4 years.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Calculate ATR(14) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX(14) for regime filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_abs = np.abs(tr)
    atr_14 = pd.Series(tr_abs).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile (20-period) for adaptive k
    atr_pct = pd.Series(atr).rolling(window=20, min_periods=10).rank(pct=True).values
    k = 1.0 + (atr_pct * 1.0)  # k from 1.0 (low vol) to 2.0 (high vol)
    
    # Calculate breakout bands
    upper_band = close + k * atr
    lower_band = close - k * atr
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.3 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14), ADX(14), volume MA(20)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(k[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx[i] > 25
        is_uptrend = is_trending and (plus_di[i] > minus_di[i])
        is_downtrend = is_trending and (plus_di[i] < minus_di[i])
        
        if position == 0:
            # Long setup: price breaks above upper band + uptrend regime + volume confirmation
            long_setup = (close[i] > upper_band[i]) and is_uptrend and volume_confirm[i]
            
            # Short setup: price breaks below lower band + downtrend regime + volume confirmation
            short_setup = (close[i] < lower_band[i]) and is_downtrend and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below lower band OR trend turns non-uptrend
            if (close[i] < lower_band[i]) or (not is_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above upper band OR trend turns non-downtrend
            if (close[i] > upper_band[i]) or (not is_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_AdaptiveVolatilityBreakout_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0