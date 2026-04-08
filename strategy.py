#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_vol_v2
# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian upper, price > 12h EMA50, and volume > 1.5x 20-period MA.
# Short when price breaks below Donchian lower, price < 12h EMA50, and volume > 1.5x 20-period MA.
# Exit when price crosses back inside Donchian channel or EMA condition fails.
# Designed for 20-40 trades/year to avoid fee drag. Works in bull/bear via trend-following with strong filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i - donchian_len + 1:i + 1])
        lower[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma = np.full(n, np.nan)
    vol_len = 20
    for i in range(vol_len - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_len + 1:i + 1])
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_len, vol_len, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend and volume filters
        price_above_ema = close[i] > ema50_12h_aligned[i]
        price_below_ema = close[i] < ema50_12h_aligned[i]
        vol_confirm = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit: price crosses back inside Donchian or EMA condition fails
            if close[i] < upper[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back inside Donchian or EMA condition fails
            if close[i] > lower[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper, above EMA, with volume confirmation
            if close[i] > upper[i] and price_above_ema and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower, below EMA, with volume confirmation
            elif close[i] < lower[i] and price_below_ema and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals