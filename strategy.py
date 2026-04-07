#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d regime filter for reduced whipsaw
# Uses 4h EMA(21) for trend direction, 1d ADX(14) for trend strength filter,
# and 1h RSI(14) for entry timing. Designed for low trade frequency (target: 15-37/year)
# by requiring multi-timeframe alignment. Works in bull markets via trend following
# and in bear markets via reduced exposure during weak trends.

name = "mtf_ema_rsi_adx_filter_1h_4h1d_v1"
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
    
    # 4h EMA(21) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = np.where((plus_di + minus_di) != 0, 100 * abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]
        
        # Trend strength filter: only trade when ADX > 25
        strong_trend = adx_1d_aligned[i] > 25
        
        # RSI conditions for entry timing
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        rsi_neutral = 40 <= rsi_1h[i] <= 60
        
        # Long conditions: bullish trend + strong trend + RSI pullback
        if bullish_trend and strong_trend and rsi_oversold:
            signals[i] = 0.20
        # Short conditions: bearish trend + strong trend + RSI bounce
        elif bearish_trend and strong_trend and rsi_overbought:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals