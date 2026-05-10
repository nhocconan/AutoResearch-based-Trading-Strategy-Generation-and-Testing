#!/usr/bin/env python3
# 4H_Donchian_20_Breakout_Trend_Volume_v2
# Hypothesis: Breakout from 20-period Donchian channel with trend (ADX>25) and volume confirmation (>1.5x average).
# Long when: price breaks above Donchian high + uptrend + volume spike.
# Short when: price breaks below Donchian low + downtrend + volume spike.
# Uses 1d trend filter: only trade in direction of daily EMA50 trend to avoid counter-trend trades.
# Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.
# Works in bull/bear by following trend and using volume to confirm institutional interest.

name = "4H_Donchian_20_Breakout_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # ADX for trend strength (14-period)
    # +DM and -DM
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment (since we lost first bar in calculations)
    donchian_high = np.concatenate([np.full(1, np.nan), donchian_high])
    donchian_low = np.concatenate([np.full(1, np.nan), donchian_low])
    adx = np.concatenate([np.full(1, np.nan), adx])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        strong_trend = adx[i] > 25
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + strong 4h trend + breakout + volume
            if daily_up and strong_trend and volume_confirm:
                if close[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + strong 4h trend + breakdown + volume
            elif daily_down and strong_trend and volume_confirm:
                if close[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: trend weakens or breakdown below Donchian low
            if not daily_up or not strong_trend or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend weakens or breakout above Donchian high
            if not daily_down or not strong_trend or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals