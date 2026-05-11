#!/usr/bin/env python3
name = "1h_RSI_Trend_Filter_4h1d"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_4h = close_4h > ema50_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Get 1d data for regime filter (ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily data
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_trending = adx > 25
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    # RSI on 1h for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.5 * vol_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_4h_aligned[i]) or np.isnan(adx_trending_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma24[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 40 (pullback in uptrend) + 4h uptrend + trending regime + volume + session
            if (rsi[i] < 40 and trend_up_4h_aligned[i] and adx_trending_aligned[i] and 
                volume_filter[i] and session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 60 (pullback in downtrend) + 4h downtrend + trending regime + volume + session
            elif (rsi[i] > 60 and not trend_up_4h_aligned[i] and adx_trending_aligned[i] and 
                  volume_filter[i] and session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 (overbought) OR 4h trend turns down
            if rsi[i] > 60 or not trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 40 (oversold) OR 4h trend turns up
            if rsi[i] < 40 or trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals