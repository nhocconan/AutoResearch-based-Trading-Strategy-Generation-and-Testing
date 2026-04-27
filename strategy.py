#!/usr/bin/env python3
"""
Hypothesis: Daily 4-week Donchian breakout with weekly trend filter and volume confirmation.
Breakouts above 4-week high (long) or below 4-week low (short) only when weekly EMA34 confirms trend direction.
Volume must exceed 1.5x daily average to confirm breakout strength.
Designed for low-frequency, high-conviction trades on 1d timeframe targeting 10-25 trades/year.
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
    
    # Calculate weekly EMA34 for trend
    wk_close = df_1w['close'].values
    ema_34_1w = np.full_like(wk_close, np.nan, dtype=np.float64)
    if len(wk_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1w[33] = np.mean(wk_close[:34])
        for i in range(34, len(wk_close)):
            ema_34_1w[i] = alpha * wk_close[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 28:  # Need 20 days for Donchian + buffer
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels (4 weeks)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = np.full_like(high_1d, np.nan, dtype=np.float64)
    donchian_low = np.full_like(low_1d, np.nan, dtype=np.float64)
    
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 20-day average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=np.float64)
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get weekly close price for trend comparison
    wk_close_price = df_1w['close'].values
    wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (34), volume MA (20)
    start_idx = max(19, 33, 19) + 1  # +1 because we need current bar's data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(wk_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Current levels
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        weekly_close = wk_close_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Breakout above 4-week high with weekly uptrend: long
            if price_now > donch_high and weekly_close > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Breakdown below 4-week low with weekly downtrend: short
            elif price_now < donch_low and weekly_close < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retouches 4-week low or trend changes
            if price_now < donch_low or weekly_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price retouches 4-week high or trend changes
            if price_now > donch_high or weekly_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_4WeekDonchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0