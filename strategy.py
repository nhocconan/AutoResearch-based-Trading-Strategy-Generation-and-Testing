#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter_VolumeSpike_v1
Hypothesis: On 12h timeframe, KAMA (adaptive trend) direction combined with RSI(14) extremes (oversold/overbought) and volume spike confirmation captures mean-reversion moves within the dominant trend. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0) to target 50-100 total trades over 4 years. Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends, avoiding sideways chop via ADX regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need for EMA34
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for stoploss calculation
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 1d data for ADX regime filter (to avoid chop)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[0], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 12h data for KAMA and RSI (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate KAMA (adaptive trend) on 12h
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h, k=1)), axis=0)  # 10-period sum of 1-period changes
    # Pad change array to match length
    change = np.concatenate([[np.nan] * 10, change])
    volatility = np.concatenate([[np.nan] * 10, volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume filter: volume > 1.8x 20-period average (tight filter for quality)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14, 10)  # EMA34 needs 34, vol MA needs 20, ATR needs 14, KAMA needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        adx_val = adx_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Get 12h close aligned for direct comparison
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        close_12h_val = close_12h_aligned[i]
        
        # Determine trend direction from 1d EMA34
        is_uptrend = close_12h_val > ema_34_val
        
        # Only trade in trending markets (ADX > 25)
        is_trending = adx_val > 25
        
        if position == 0:
            # Look for entry signals: mean reversion in direction of trend
            # Long: price below KAMA (dip) and RSI oversold (<30) in uptrend
            # Short: price above KAMA (rally) and RSI overbought (>70) in downtrend
            long_signal = (close_12h_val < kama_val) and (rsi_val < 30) and is_uptrend and vol_spike[i] and is_trending
            short_signal = (close_12h_val > kama_val) and (rsi_val > 70) and (not is_uptrend) and vol_spike[i] and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_12h_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_12h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price crosses above KAMA (trend resumes)
            # 2. RSI overbought (>70) - take profit
            # 3. ATR-based stoploss: 2.0 * ATR below entry
            exit_signal = close_12h_val > kama_val
            profit_signal = rsi_val > 70
            stop_signal = close_12h_val < (entry_price - 2.0 * atr_val)
            if exit_signal or profit_signal or stop_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price crosses below KAMA (trend resumes)
            # 2. RSI oversold (<30) - take profit
            # 3. ATR-based stoploss: 2.0 * ATR above entry
            exit_signal = close_12h_val < kama_val
            profit_signal = rsi_val < 30
            stop_signal = close_12h_val > (entry_price + 2.0 * atr_val)
            if exit_signal or profit_signal or stop_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_RSI_Filter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0