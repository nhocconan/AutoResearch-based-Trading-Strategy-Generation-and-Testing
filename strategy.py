#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R with weekly trend filter and volume confirmation.
Williams %R > -20 (overbought) or < -80 (oversold) signals reversals.
Weekly EMA(34) determines trend direction: only take Williams %R signals in trend direction.
Volume > 1.5x 20-period average confirms participation.
Designed to capture overextended moves in both bull and bear markets with controlled frequency.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    wk_close = df_1w['close'].values
    wk_ema_34 = pd.Series(wk_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    wk_ema_34_aligned = align_htf_to_ltf(prices, df_1w, wk_ema_34)
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    
    for i in range(n):
        if i < lookback - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
    
    williams_r = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(n):
        if highest_high[i] == lowest_low[i] or np.isnan(highest_high[i]):
            williams_r[i] = -50.0  # neutral when no range
        else:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R (14) + weekly EMA (34) + volume MA (20)
    start_idx = max(lookback, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(wk_ema_34_aligned[i]) or 
            np.isnan(wk_close[-1]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        wr = williams_r[i]
        price_now = close[i]
        vol_now = volume[i]
        wk_ema = wk_ema_34_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) + price above weekly EMA (uptrend) + volume
            if wr < -80 and price_now > wk_ema and vol_filter:
                signals[i] = size
                position = 1
            # Short setup: Williams %R overbought (> -20) + price below weekly EMA (downtrend) + volume
            elif wr > -20 and price_now < wk_ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend changes
            if wr > -50 or price_now < wk_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend changes
            if wr < -50 or price_now > wk_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0