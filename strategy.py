#!/usr/bin/env python3
"""
4h_HTFTrend_LTFEntry_v2
Refined: 4h trend with 1d/1w confirmation and precise entry.
- Trend filter: 1d close > 1w EMA34 (bull) or < 1w EMA34 (bear)
- Entry: Price pulls back to 4h EMA20 with volume > 1.5x average
- Exit: Trend reversal or price moves 1% away from EMA20
- Position sizing: 0.25 long/short
- Designed for ~25-40 trades/year per symbol
Works in bull (trend continuation) and bear (trend continuation) markets
"""

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
    
    # Get 1d and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d close aligned to 4h timeframe
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 4h EMA20 for entry timing
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for EMA/volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(ema_20_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d close relative to 1w EMA34
        bull_trend = close_1d_aligned[i] > ema_34_1w_aligned[i]
        bear_trend = close_1d_aligned[i] < ema_34_1w_aligned[i]
        
        # Entry condition: price near 4h EMA20 with volume
        near_ema = abs(close[i] - ema_20_4h[i]) / ema_20_4h[i] < 0.005  # within 0.5%
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        # Exit condition: price moves 1% away from EMA20
        far_from_ema = abs(close[i] - ema_20_4h[i]) / ema_20_4h[i] > 0.01
        
        if position == 0:
            # Long: bull trend + pullback to EMA + volume
            if bull_trend and near_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bear trend + pullback to EMA + volume
            elif bear_trend and near_ema and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or price moves away from EMA
            if not bull_trend or far_from_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or price moves away from EMA
            if not bear_trend or far_from_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTFTrend_LTFEntry_v2"
timeframe = "4h"
leverage = 1.0