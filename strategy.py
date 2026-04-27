#!/usr/bin/env python3
"""
Hypothesis: 12-hour Price Channel Breakout with weekly trend filter and volume confirmation.
Buy when price breaks above 12h Donchian upper channel (20-period) with weekly uptrend and volume spike.
Sell when price breaks below 12h Donchian lower channel with weekly downtrend and volume spike.
Exit when price returns to the middle of the channel or trend reverses.
Designed to work in both bull and bear markets by requiring volume confirmation and trend alignment.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels: upper=max(high,period), lower=min(low,period)"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64), np.full_like(high, np.nan, dtype=np.float64)
    
    upper = np.full_like(high, np.nan, dtype=np.float64)
    lower = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_ema(values, period):
    """Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    
    ema = np.full_like(values, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

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
    ema_34_1w = calculate_ema(wk_close, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h Donchian channels (20-period)
    upper_channel, lower_channel = calculate_donchian_channels(high, low, 20)
    middle_channel = (upper_channel + lower_channel) / 2.0
    
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
    
    # Warmup: need Donchian (20) + EMA (34) + volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        upper = upper_channel[i]
        lower = lower_channel[i]
        middle = middle_channel[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average (tighter to reduce trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper channel with weekly uptrend and volume
            if price_now > upper and ema_trend > lower and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower channel with weekly downtrend and volume
            elif price_now < lower and ema_trend < upper and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of channel or trend turns down
            if price_now < middle or ema_trend < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle of channel or trend turns up
            if price_now > middle or ema_trend > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0