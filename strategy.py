#!/usr/bin/env python3
"""
Hypothesis: Daily Bollinger Band breakout with weekly trend filter and volume confirmation.
In bull markets: long when price breaks above upper BB with weekly uptrend and high volume.
In bear markets: short when price breaks below lower BB with weekly downtrend and high volume.
Uses Bollinger Bands (20,2) for volatility breakouts, weekly EMA50 for trend filter,
and volume > 1.5x 20-day average for confirmation. Designed for 1d timeframe to
capture multi-day trends while minimizing trade frequency (<20 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, length=20, std_dev=2.0):
    """Calculate Bollinger Bands: upper, middle (SMA), lower"""
    if len(length) < length:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    sma = pd.Series(close).rolling(window=length, min_periods=length).mean().values
    std = pd.Series(close).rolling(window=length, min_periods=length).std().values
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    wk_close = df_1w['close'].values
    ema_50_1w = calculate_ema(wk_close, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Bollinger Bands (20,2)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close, 20, 2.0)
    
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
    
    # Warmup: need BB (20) + EMA (50) + volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        bb_upper = upper_bb[i]
        bb_lower = lower_bb[i]
        bb_middle = middle_bb[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: weekly close above/below weekly EMA50
        wk_close_price = df_1w['close'].values
        wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
        if np.isnan(wk_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_close = wk_close_aligned[i]
        
        if position == 0:
            # Bullish breakout: price above upper BB with weekly uptrend and high volume
            if price_now > bb_upper and weekly_close > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bearish breakout: price below lower BB with weekly downtrend and high volume
            elif price_now < bb_lower and weekly_close < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle BB or trend changes
            if price_now < bb_middle or weekly_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle BB or trend changes
            if price_now > bb_middle or weekly_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0