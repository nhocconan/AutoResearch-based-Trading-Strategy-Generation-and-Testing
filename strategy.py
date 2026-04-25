#!/usr/bin/env python3
"""
1h_Donchian_Breakout_VolumeRegime_4hTrend_v1
Hypothesis: Trade 1h Donchian(20) breakouts in direction of 4h EMA50 trend with volume spike confirmation and chop regime filter during UTC 08-20 session.
In bull markets: trend continuation via breakouts. In bear markets: mean reversion via failed breakouts in choppy regimes.
Volume spike confirms institutional interest. Chop filter avoids whipsaws. Target: 15-35 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for HTF filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA34 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1h ATR(14) for Donchian and chop filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Calculate 1h chopiness index (14-period) for regime filter
    def calculate_chop(high, low, close, window=14):
        if len(high) < window:
            return np.full(len(high), np.nan)
        atr_sum = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        for i in range(window, len(high)+1):
            atr_sum[i-1] = np.sum(tr[i-window+1:i+1])
        max_min_range = np.zeros(len(high))
        for i in range(window-1, len(high)):
            max_min_range[i] = np.max(high[i-window+1:i+1]) - np.min(low[i-window+1:i+1])
        chop = np.where(max_min_range > 0, 100 * np.log10(atr_sum / max_min_range) / np.log10(window), 50.0)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(lookback, 50, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine regime from chop (chop > 61.8 = ranging, chop < 38.2 = trending)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # Determine 4h and 1d trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)[i]
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(close_4h_aligned) or np.isnan(close_1d_aligned):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        htf_4h_bullish = close_4h_aligned > ema_50_4h_aligned[i]
        htf_1d_bullish = close_1d_aligned > ema_34_1d_aligned[i]
        
        # Volume confirmation: need significant spike
        volume_confirmed = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long setup: Donchian breakout above resistance + 4h bullish trend + volume + NOT choppy
            long_setup = (close[i] > highest_high[i]) and htf_4h_bullish and volume_confirmed and not is_choppy
            
            # Short setup: Donchian breakout below support + 4h bearish trend + volume + NOT choppy
            short_setup = (close[i] < lowest_low[i]) and (not htf_4h_bullish) and volume_confirmed and not is_choppy
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: Donchian breakdown below support OR 1d trend turns bearish OR chop increases significantly
            if (close[i] < lowest_low[i]) or (not htf_1d_bullish) or (chop[i] > 50.0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: Donchian breakout above resistance OR 1d trend turns bullish OR chop increases significantly
            if (close[i] > highest_high[i]) or (htf_1d_bullish) or (chop[i] > 50.0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Donchian_Breakout_VolumeRegime_4hTrend_v1"
timeframe = "1h"
leverage = 1.0