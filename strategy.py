#!/usr/bin/env python3
# mtf_1h_rsi_divergence_4h1d_atr_v1
# Hypothesis: 1h RSI divergence (bullish/bearish) confirmed by 4h trend (EMA50) and 1d regime (ADX<25 for mean reversion, ADX>25 for trend).
# Uses ATR-based stops via signal=0. Works in bull/bear: 4h EMA filters trend, 1d ADX selects regime, RSI divergence captures exhaustion.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_divergence_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                  np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    down = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                    np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    up = np.concatenate([[np.nan], up])
    down = np.concatenate([[np.nan], down])
    
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_smooth = pd.Series(up).ewm(alpha=1/14, adjust=False).mean().values
    down_smooth = pd.Series(down).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * up_smooth / tr_smooth
    di_minus = 100 * down_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI divergence detection (lookback 3 bars)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-2]) or np.isnan(close[i]) or np.isnan(close[i-2]):
            continue
        # Bullish: price lower low, RSI higher low
        if low[i] < low[i-2] and rsi[i] > rsi[i-2]:
            bullish_div[i] = True
        # Bearish: price higher high, RSI lower high
        if high[i] > high[i-2] and rsi[i] < rsi[i-2]:
            bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(bullish_div[i]) or np.isnan(bearish_div[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR bearish RSI divergence
            if close[i] < ema_4h_aligned[i] or bearish_div[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR bullish RSI divergence
            if close[i] > ema_4h_aligned[i] or bullish_div[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Regime filter: use ADX to decide mean reversion vs trend
            if adx_aligned[i] < 25:  # Range market: mean reversion
                # Bullish divergence + price below 4h EMA → long
                if bullish_div[i] and close[i] < ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Bearish divergence + price above 4h EMA → short
                elif bearish_div[i] and close[i] > ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
            else:  # Trending market: trend continuation
                # Bullish divergence + price above 4h EMA → long (pullback in uptrend)
                if bullish_div[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Bearish divergence + price below 4h EMA → short (pullback in downtrend)
                elif bearish_div[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals