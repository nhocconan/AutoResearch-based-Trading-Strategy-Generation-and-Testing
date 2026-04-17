#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trading with 4h directional bias and 1d trend filter.
# Uses 4h EMA21 for trend direction (price > EMA21 = long bias, < EMA21 = short bias)
# 1d ADX > 25 ensures we only trade in strong trending markets to avoid whipsaws
# Entry triggered on 1h when price crosses EMA13 in direction of 4h trend
# Conservative position size (0.20) and session filter (08-20 UTC) to limit trades to ~20-40/year
# Designed to work in both bull (trend following) and bear (avoids false signals via ADX filter)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h EMA21 for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 1d ADX (14-period) for trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di_1d = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1h EMA13 for entry timing ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Pre-compute session hours (08-20 UTC)
    if isinstance(prices.index, pd.DatetimeIndex):
        hours = prices.index.hour
    else:
        hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_13[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filters
        strong_trend = adx_1d_aligned[i] > 25
        bullish_bias = close[i] > ema_21_4h_aligned[i]  # price above 4h EMA21
        bearish_bias = close[i] < ema_21_4h_aligned[i]  # price below 4h EMA21
        
        # Entry logic: only enter when flat
        if position == 0:
            if strong_trend and in_session:
                # Go long if bullish bias and price crosses above EMA13
                if bullish_bias and close[i] > ema_13[i] and (i == warmup or close[i-1] <= ema_13[i-1]):
                    signals[i] = 0.20
                    position = 1
                    continue
                # Go short if bearish bias and price crosses below EMA13
                elif bearish_bias and close[i] < ema_13[i] and (i == warmup or close[i-1] >= ema_13[i-1]):
                    signals[i] = -0.20
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long if bearish bias forms or trend weakens
            if bearish_bias or not strong_trend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if bullish bias forms or trend weakens
            if bullish_bias or not strong_trend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA13_4hEMA21_1dADXFilter"
timeframe = "1h"
leverage = 1.0