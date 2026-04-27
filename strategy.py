#!/usr/bin/env python3
"""
Hypothesis: Daily Supertrend(ATR=10, mult=3) with weekly trend filter (EMA50) and volume confirmation.
Trades in direction of weekly trend when price crosses Supertrend on daily timeframe with above-average volume.
Weekly trend filter avoids counter-trend trades in strong trends; volume confirms breakout strength.
Designed for low frequency: ~10-20 trades/year per symbol to minimize fee drag in ranging/bear markets.
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
    
    # Get daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate ATR(10)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_1d + low_1d) / 2
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume MA(20)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup periods
    start_idx = max(10, 20, 50)  # ATR, volume MA, weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        st_now = supertrend_aligned[i]
        dir_now = direction_aligned[i]
        weekly_trend = ema_50_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Supertrend crossover with volume and weekly trend alignment
        if position == 0:
            # Long: price crosses above Supertrend with volume + weekly uptrend
            if price_now > st_now and price_now <= close[i-1] and vol_now > volume[i-1] and vol_filter and weekly_trend > close_1w[-1] if len(close_1w) > 0 else True:
                # Simplified: price above Supertrend and weekly EMA above previous weekly close (uptrend)
                if price_now > st_now and weekly_trend > np.mean(close_1w[-5:]) if len(close_1w) >= 5 else weekly_trend > close_1w[0]:
                    signals[i] = size
                    position = 1
            # Short: price crosses below Supertrend with volume + weekly downtrend
            elif price_now < st_now and price_now >= close[i-1] and vol_now < volume[i-1] and vol_filter and weekly_trend < np.mean(close_1w[-5:]) if len(close_1w) >= 5 else weekly_trend < close_1w[0]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Supertrend
            if price_now < st_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Supertrend
            if price_now > st_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Supertrend_10x3_VolumeFilter_1wEMA50"
timeframe = "1d"
leverage = 1.0