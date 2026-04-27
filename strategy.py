#!/usr/bin/env python3
"""
Hypothesis: 4-hour 20-period Donchian breakout with 1-day ATR volatility filter and 1-day EMA34 trend filter.
In bull market (price > 1-day EMA34): long when price breaks above 20-bar high and ATR(14) < 0.8 * ATR(50).
In bear market (price < 1-day EMA34): short when price breaks below 20-bar low and ATR(14) < 0.8 * ATR(50).
Breakouts in low volatility conditions are more likely to sustain, while trend filter ensures direction alignment.
Target: 20-40 trades/year per symbol.
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
    
    # Get daily data for trend filter and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR for volatility filters
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_vals = df_1d['close'].values
    
    # True Range components
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close_vals[0]], daily_close_vals[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close_vals[0]], daily_close_vals[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR14 and ATR50
    atr14 = np.full_like(tr, np.nan)
    atr50 = np.full_like(tr, np.nan)
    
    # ATR14 calculation
    if len(tr) >= 14:
        atr14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr14[i] = (13/14) * atr14[i-1] + (1/14) * tr[i]
    
    # ATR50 calculation
    if len(tr) >= 50:
        atr50[49] = np.mean(tr[:50])
        for i in range(50, len(tr)):
            atr50[i] = (49/50) * atr50[i-1] + (1/50) * tr[i]
    
    # ATR ratio: ATR14/ATR50 < 0.8 indicates low volatility
    atr_ratio = np.full_like(tr, np.nan)
    valid = (~np.isnan(atr14)) & (~np.isnan(atr50)) & (atr50 > 0)
    atr_ratio[valid] = atr14[valid] / atr50[valid]
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Daily close price for trend comparison
    daily_close_price = df_1d['close'].values
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ATR ratio (50), daily EMA34 (34)
    start_idx = max(19, 50, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(daily_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price
        price_now = close[i]
        
        # Current indicators
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol_filter = atr_ratio_aligned[i] < 0.8  # Low volatility filter
        ema_trend = ema_34_1d_aligned[i]
        daily_close_val = daily_close_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Bull market: long on breakout above Donchian high in low vol
            if price_now > donchian_high and daily_close_val > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear market: short on breakout below Donchian low in low vol
            elif price_now < donchian_low and daily_close_val < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend turns bear
            if price_now < donchian_low or daily_close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend turns bull
            if price_now > donchian_high or daily_close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_ATRVolatilityFilter_DailyTrend"
timeframe = "4h"
leverage = 1.0