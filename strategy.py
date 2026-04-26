#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: On daily timeframe, enter long when KAMA trend is up (close > KAMA) AND RSI < 30 (oversold) AND choppiness regime is trending (CHOP < 38.2). Enter short when KAMA trend is down (close < KAMA) AND RSI > 70 (overbought) AND CHOP < 38.2. Exit when trend reverses. Uses 1w EMA200 as higher timeframe trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) with strong edge in both bull and bear regimes via mean reversion in trending markets.
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
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation: ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    direction = np.abs(close_series - close_series.shift(10))
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.nan_to_num(sc, nan=0.0)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index (CHOP)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(atr14 / (hh14 - ll14)) / log10(14)
    range_14 = hh14 - ll14
    chop = 100 * np.log10(atr14 / np.maximum(range_14, 1e-10)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA, RSI, CHOP, and EMA200 warmup
    start_idx = max(100, 14, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend and regime conditions
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        chop_trending = chop[i] < 38.2  # trending market
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold (<30) + trending regime + 1w uptrend filter
            long_signal = kama_uptrend and (rsi[i] < 30) and chop_trending and (close[i] > ema_200_1w_aligned[i])
            
            # Short: KAMA downtrend + RSI overbought (>70) + trending regime + 1w downtrend filter
            short_signal = kama_downtrend and (rsi[i] > 70) and chop_trending and (close[i] < ema_200_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA downtrend OR chop becomes ranging (>61.8) OR 1w trend fails
            if (not kama_uptrend) or (chop[i] > 61.8) or (close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA uptrend OR chop becomes ranging (>61.8) OR 1w trend fails
            if (not kama_downtrend) or (chop[i] > 61.8) or (close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0