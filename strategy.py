#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA21 pullback strategy with 4h EMA50 trend filter and 1d ADX regime filter.
# Long when: price > 1h EMA21 AND 4h EMA50 > 1d EMA50 (uptrend) AND 1d ADX > 25 (strong trend)
# Short when: price < 1h EMA21 AND 4h EMA50 < 1d EMA50 (downtrend) AND 1d ADX > 25 (strong trend)
# Exit when price crosses 1h EMA21 in opposite direction.
# Uses discrete position size 0.20 to limit drawdown in bear markets.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.
# Works in bull (trend following) and bear (strong trend filters avoid whipsaws).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data once before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data once before loop for EMA50 and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h EMA21 for entry/exit
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_50_4h_val = ema_50_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        adx_val = adx_aligned[i]
        ema_21_val = ema_21[i]
        price = close[i]
        
        # Trend alignment: 4h EMA50 vs 1d EMA50
        uptrend = ema_50_4h_val > ema_50_1d_val
        downtrend = ema_50_4h_val < ema_50_1d_val
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_val > 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 1h EMA21
            if price < ema_21_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 1h EMA21
            if price > ema_21_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > 1h EMA21 AND 4h EMA50 > 1d EMA50 (uptrend) AND strong trend
            if price > ema_21_val and uptrend and strong_trend:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price < 1h EMA21 AND 4h EMA50 < 1d EMA50 (downtrend) AND strong trend
            elif price < ema_21_val and downtrend and strong_trend:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA21_Pullback_4hEMA50_Trend_1dADX25_V1"
timeframe = "1h"
leverage = 1.0