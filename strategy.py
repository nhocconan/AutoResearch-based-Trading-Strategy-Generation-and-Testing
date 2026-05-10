#!/usr/bin/env python3
"""
1d_Keltner_Breakout_Volume_Squeeze
Hypothesis: Daily Keltner channel breakout with volume squeeze filter and weekly trend filter.
Works in bull/bear markets by using volatility-based breakouts (Keltner) combined with volume confirmation
and weekly trend alignment to filter counter-trend moves. Target: 15-25 trades/year to minimize fee drag.
"""

name = "1d_Keltner_Breakout_Volume_Squeeze"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily ATR for Keltner channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) for Keltner channels
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Daily EMA20 for Keltner center
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: EMA20 ± 2 * ATR
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Volume filter: volume > 1.5x 20-day EMA of volume
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    # Volatility squeeze filter: ATR(10) < ATR(50) indicates low volatility
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_squeeze = atr < atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ATR(50) and EMA20
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Keltner band with weekly uptrend, volume, and volatility squeeze
            if (close[i] > keltner_upper[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_filter[i] and 
                volatility_squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner band with weekly downtrend, volume, and volatility squeeze
            elif (close[i] < keltner_lower[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_filter[i] and 
                  volatility_squeeze[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below EMA20 or weekly trend change
            if close[i] < ema_20[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above EMA20 or weekly trend change
            if close[i] > ema_20[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals