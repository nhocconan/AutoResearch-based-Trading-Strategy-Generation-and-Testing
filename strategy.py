#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA adapts to market noise, providing a reliable trend signal. 
Combined with RSI (for momentum) and Choppiness Index (to avoid ranging markets), 
this strategy captures strong trends while avoiding whipsaws in consolidation. 
Designed for low trade frequency (10-20/year) on daily timeframe to minimize 
fee drag and perform well in both bull and bear markets.
"""

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA (Kaufman Adaptive Moving Average) - 10 periods
    # Efficiency Ratio (ER) = |Change| / Volatility
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily (already on daily, but ensure proper alignment)
    kama_aligned = kama  # No need to align as it's calculated on close
    
    # RSI (14 periods)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (14 periods)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = hh - ll
    chop = np.where(hh_ll == 0, 50, 100 * np.log10(atr_sum / hh_ll) / np.log10(14))
    # Pad first 14 values
    chop = np.concatenate([np.full(14, np.nan), chop[14:]])
    
    # 1w trend filter: EMA(34) on close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Conditions
    # KAMA direction: price above KAMA = uptrend, below = downtrend
    kama_up = close > kama_aligned
    kama_down = close < kama_aligned
    
    # RSI conditions: avoid extremes, look for momentum
    rsi_bullish = (rsi > 50) & (rsi < 70)  # Not overbought
    rsi_bearish = (rsi < 50) & (rsi > 30)  # Not oversold
    
    # Choppiness filter: only trade when trending (CHOP < 38.2) or strong mean reversion (CHOP > 61.8)
    chop_trending = chop < 38.2
    chop_ranging = chop > 61.8
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            continue
            
        # LONG: KAMA up + RSI bullish + trending market
        if kama_up[i] and rsi_bullish[i] and chop_trending[i]:
            signals[i] = 0.25
        # SHORT: KAMA down + RSI bearish + trending market
        elif kama_down[i] and rsi_bearish[i] and chop_trending[i]:
            signals[i] = -0.25
        # Optional: mean reversion in ranging markets (commented out to reduce trades)
        # elif kama_down[i] and rsi > 70 and chop_ranging[i]:  # Overbought in range
        #     signals[i] = -0.20
        # elif kama_up[i] and rsi < 30 and chop_ranging[i]:   # Oversold in range
        #     signals[i] = 0.20
        else:
            signals[i] = 0.0
    
    return signals