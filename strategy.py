#!/usr/bin/env python3
"""
1h_4h1d_TrendPullback_MeanReversion
Hypothesis: In trending markets (identified by 4h EMA alignment and 1d ADX), 
price pulls back to the 20-period EMA on 1h, offering mean-reversion entries.
In ranging markets (low 1d ADX), fade moves outside Bollinger Bands.
Uses 4h for trend direction, 1d for regime (ADX), 1h for entry timing.
Target: 15-35 trades/year via strict confluence.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Exponential Moving Average."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    sma = np.mean(values[:period])
    ema_val = np.full(len(values), np.nan)
    ema_val[period-1] = sma
    multiplier = 2 / (period + 1)
    for i in range(period, len(values)):
        ema_val[i] = (values[i] * multiplier) + (ema_val[i-1] * (1 - multiplier))
    return ema_val

def rsi(close, period=14):
    """Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def adx(high, low, close, period=14):
    """Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    tr = np.zeros(len(high))
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
        minus_dm[i] = max(high[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / 
                     pd.Series(tr).ewm(alpha=1/period, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / 
                      pd.Series(tr).ewm(alpha=1/period, adjust=False).mean())
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_val = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx_val

def bollinger_bands(close, period=20, std_dev=2):
    """Bollinger Bands."""
    if len(close) < period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower, sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend EMA
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = ema(close_4h, 20)
    ema50_4h = ema(close_4h, 50)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for regime (ADX) and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = adx(high_1d, low_1d, close_1d, 14)
    bb_upper, bb_lower, bb_middle = bollinger_bands(close_1d, 20, 2)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    
    # 1h indicators
    ema20_1h = ema(close, 20)
    rsi_1h = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(ema20_1h[i]) or
            np.isnan(rsi_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend (4h EMA alignment)
        uptrend_4h = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend_4h = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        
        # Determine regime (1d ADX)
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Price relative to 1h EMA20
        price_vs_ema = close[i] - ema20_1h[i]
        
        if position == 0:
            # Long conditions
            if trending and uptrend_4h:
                # In uptrend: buy pullbacks to EMA20
                if price_vs_ema <= 0 and rsi_1h[i] < 40:  # slight pullback
                    signals[i] = 0.20
                    position = 1
            elif ranging:
                # In range: buy at support (BB lower)
                if close[i] <= bb_lower_aligned[i]:
                    signals[i] = 0.20
                    position = 1
            
            # Short conditions
            elif trending and downtrend_4h:
                # In downtrend: sell rallies to EMA20
                if price_vs_ema >= 0 and rsi_1h[i] > 60:  # slight rally
                    signals[i] = -0.20
                    position = -1
            elif ranging:
                # In range: sell at resistance (BB upper)
                if close[i] >= bb_upper_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: reverse conditions or opposite signal
            if (trending and downtrend_4h and price_vs_ema >= 0 and rsi_1h[i] > 60) or \
               (ranging and close[i] >= bb_middle_aligned[i]) or \
               (not trending and not ranging and rsi_1h[i] > 70):  # overbought
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: reverse conditions or opposite signal
            if (trending and uptrend_4h and price_vs_ema <= 0 and rsi_1h[i] < 40) or \
               (ranging and close[i] <= bb_middle_aligned[i]) or \
               (not trending and not ranging and rsi_1h[i] < 30):  # oversold
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_TrendPullback_MeanReversion"
timeframe = "1h"
leverage = 1.0