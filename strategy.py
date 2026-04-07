#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour trend filter and 1-day regime filter
# Uses 4h EMA crossover for trend direction, 1d ADX for regime (trending vs ranging),
# and 1h RSI for entry timing. Designed to work in both bull and bear markets
# by avoiding trades in low-volatility ranging regimes. Target: 60-150 total trades.
# Session filter (08-20 UTC) reduces noise. Position size: 0.20.

name = "1h_momentum_4h_trend_1d_regime_v1"
timeframe = "1h"
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
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1-day data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(25) and EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_25_4h = pd.Series(close_4h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_25_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(np.isnan(rs), 0, rs)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_25_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA(25) > EMA(50) = uptrend, < = downtrend
        uptrend = ema_25_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend = ema_25_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Regime filter: ADX > 25 = trending market (good for momentum)
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:  # flat - look for entry
            # Long: uptrend + trending + RSI < 30 (oversold bounce)
            if uptrend and trending and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + trending + RSI > 70 (overbought bounce)
            elif downtrend and trending and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        elif position == 1:  # long position
            # Exit: RSI > 70 (overbought) or trend change
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or trend change
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals