#!/usr/bin/env python3
"""
Hypothesis: 4-hour RSI(14) extreme reversal with 12-hour volume confirmation and 12-hour trend filter.
Trades RSI extremes (<30 for long, >70 for short) when 12h volume exceeds average and 12h trend aligns.
Designed to work in both bull and bear markets by using 12h trend as filter and volume to confirm reversal strength.
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
    
    # Get 4-hour data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align RSI to 4-hour timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Get 12-hour data for volume filter and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 12-hour EMA(25) for trend
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need RSI, volume MA, and 12h EMA
    start_idx = max(14, 20, 25)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(ema_25_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_12h = ema_25_12h_aligned[i]
        
        # Current RSI
        rsi_now = rsi_aligned[i]
        
        # Volume filter: volume > 1.5x 12-hour average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: RSI extreme with volume and 12h trend alignment
        if position == 0:
            # Long: RSI < 30 (oversold) with volume + 12h uptrend
            if rsi_now < 30 and vol_filter and price_now > trend_12h:
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) with volume + 12h downtrend
            elif rsi_now > 70 and vol_filter and price_now < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or price breaks below 12h trend
            if rsi_now > 50 or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 or price breaks above 12h trend
            if rsi_now < 50 or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSIExtreme_12hVolume_12hTrend"
timeframe = "4h"
leverage = 1.0