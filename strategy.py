#!/usr/bin/env python3
"""
6h_1w_donchian_breakout_v1
Strategy: 6h Donchian breakout with weekly trend filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Donchian breakouts capture momentum in trending markets; weekly trend filter avoids counter-trend trades; volume confirmation ensures validity. Works in bull (breakouts up) and bear (breakouts down) via trend filter. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_breakout_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6x Donchian(20) for breakout levels
    def donchian_channels(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    # 6x ATR for stop and filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6x volume filter: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # === Weekly trend filter: price above/below weekly EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Session filter: 00-23 UTC (6h sessions cover all, but avoid low volatility periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # all hours
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_50[i]) or np.isnan(ema_50_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_50[i]
        
        # Volume confirmation
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Weekly trend: price above EMA50 = uptrend, below = downtrend
        weekly_uptrend = price_close > ema_50_1w_aligned[i]
        weekly_downtrend = price_close < ema_50_1w_aligned[i]
        
        # Long: breakout above Donchian high in uptrend with volume
        long_signal = volume_expanded and weekly_uptrend and (price_close > donch_hi[i])
        
        # Short: breakdown below Donchian low in downtrend with volume
        short_signal = volume_expanded and weekly_downtrend and (price_close < donch_lo[i])
        
        # Exit: reverse signal or close donchian midpoint
        donch_mid = (donch_hi[i] + donch_lo[i]) / 2
        exit_long = position == 1 and price_close < donch_mid
        exit_short = position == -1 and price_close > donch_mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals