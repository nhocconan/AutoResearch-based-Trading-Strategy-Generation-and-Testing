#!/usr/bin/env python3
"""
Hypothesis: 6-hour Heikin-Ashi RSI(14) with weekly trend filter and daily volume confirmation.
Uses Heikin-Ashi candles to reduce noise, RSI for mean reversion, weekly trend for direction,
and daily volume to confirm momentum. Designed to work in both bull and bear markets by
filtering trades with higher timeframe trend and requiring volume confirmation to avoid false signals.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
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
    
    # Calculate Heikin-Ashi close
    ha_close = (high + low + close + close) / 4.0  # Simplified: using current close as prev close proxy
    # For better accuracy, we need previous HA close, but we'll use this approximation for signal generation
    # In practice, HA close = (open+high+low+close)/4, but we avoid lookback by using current values
    
    # Calculate RSI(14) on Heikin-Ashi close
    delta = np.diff(ha_close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on daily
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need RSI, weekly EMA, and daily volume MA
    start_idx = max(14, 50, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        ha_close_now = ha_close[i]
        rsi_now = rsi[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: RSI mean reversion with weekly trend and volume confirmation
        if position == 0:
            # Long: RSI oversold (≤30) with weekly uptrend and volume
            if rsi_now <= 30 and ha_close_now > weekly_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (≥70) with weekly downtrend and volume
            elif rsi_now >= 70 and ha_close_now < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought (≥70) or price crosses below weekly EMA
            if rsi_now >= 70 or ha_close_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold (≤30) or price crosses above weekly EMA
            if rsi_now <= 30 or ha_close_now > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_HA_RSI14_WeeklyTrend_DailyVolume"
timeframe = "6h"
leverage = 1.0