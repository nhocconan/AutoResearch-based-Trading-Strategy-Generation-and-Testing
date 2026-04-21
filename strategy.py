#!/usr/bin/env python3
"""
1h_HTF4d_Regime_Adaptive_MeanReversion_V1
Hypothesis: On 1h timeframe, use 4h HTF for regime detection (ADX) and 1d HTF for trend bias (EMA50). 
In ranging markets (ADX<25): mean reversion at Bollinger Bands (20,2) with RSI confirmation. 
In trending markets (ADX>25): pullback to EMA21 in direction of 1d trend. 
Session filter (08-20 UTC) to avoid low-liquidity periods. 
Target: 15-30 trades/year per symbol via tight entry conditions and regime filters.
Uses discrete position sizing (0.20) to minimize fee churn. Designed to work in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop ===
    # 4h for ADX regime detection
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 1d for trend bias (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Calculate 4h ADX (14-period) for regime detection ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                                 np.maximum(high_4h - np.roll(high_4h, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                                  np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0))
    
    # Smoothed values
    tr_14 = tr_4h.rolling(window=14, min_periods=14).mean()
    dm_plus_14 = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_14 = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_4h = adx.values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === Calculate 1d EMA50 for trend bias ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate 1h indicators for entry timing ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Volume confirmation (1.5x 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(rsi[i]) or
            np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]
        
        adx_val = adx_4h_aligned[i]
        is_ranging = adx_val < 25
        is_trending = adx_val > 25
        
        if position == 0:
            # Long entry conditions
            long_signal = False
            if is_ranging:
                # Mean reversion: price at lower BB + RSI oversold
                long_signal = (price <= lower_bb[i]) and (rsi[i] < 30) and vol_ok
            elif is_trending:
                # Pullback: price near EMA21 + above 1d EMA50 (uptrend bias)
                long_signal = (price >= ema_21[i] * 0.998) and (price <= ema_21[i] * 1.002) and \
                              (price > ema_50_1d_aligned[i]) and vol_ok
            
            # Short entry conditions
            short_signal = False
            if is_ranging:
                # Mean reversion: price at upper BB + RSI overbought
                short_signal = (price >= upper_bb[i]) and (rsi[i] > 70) and vol_ok
            elif is_trending:
                # Pullback: price near EMA21 + below 1d EMA50 (downtrend bias)
                short_signal = (price >= ema_21[i] * 0.998) and (price <= ema_21[i] * 1.002) and \
                               (price < ema_50_1d_aligned[i]) and vol_ok
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions
            exit_signal = False
            if is_ranging:
                # Exit mean reversion: price at middle BB or RSI neutral
                exit_signal = (price >= sma_20[i]) or (rsi[i] > 50)
            else:  # trending
                # Exit trend: price breaks below EMA21 or loss of momentum
                exit_signal = (price < ema_21[i] * 0.99) or (rsi[i] < 40)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions
            exit_signal = False
            if is_ranging:
                # Exit mean reversion: price at middle BB or RSI neutral
                exit_signal = (price <= sma_20[i]) or (rsi[i] < 50)
            else:  # trending
                # Exit trend: price breaks above EMA21 or loss of momentum
                exit_signal = (price > ema_21[i] * 1.01) or (rsi[i] > 60)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HTF4d_Regime_Adaptive_MeanReversion_V1"
timeframe = "1h"
leverage = 1.0