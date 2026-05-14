#!/usr/bin/env python3
"""
1h EMA20 Pullback with 4h EMA50 Trend and Volume Confirmation
Hypothesis: In strong trends (4h EMA50 direction), 1h EMA20 pullbacks offer high-probability entries.
Volume confirmation filters weak breakouts. Works in bull markets via trend-following pullbacks
and in bear markets via shorting trend-following pullbacks. Discrete sizing (0.20) limits drawdown.
Target: 60-120 total trades over 4 years (15-30/year) on 1h timeframe.
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
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h EMA20 for pullback entries
    ema_20 = np.full(n, np.nan)
    if len(close) >= 20:
        ema_20_series = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
        ema_20 = ema_20_series.values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_4h, EMA20, and volume MA to propagate
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema50_4h = ema_50_4h_aligned[i]
        ema20 = ema_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above EMA20 AND 4h EMA50 uptrend (close > EMA50) AND volume confirmation
            long_condition = (curr_close > ema20) and (curr_close > ema50_4h) and volume_confirm
            # Short: price below EMA20 AND 4h EMA50 downtrend (close < EMA50) AND volume confirmation
            short_condition = (curr_close < ema20) and (curr_close < ema50_4h) and volume_confirm
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price closes below EMA20 (trend invalidation)
            if curr_close < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above EMA20 (trend invalidation)
            if curr_close > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_Pullback_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0