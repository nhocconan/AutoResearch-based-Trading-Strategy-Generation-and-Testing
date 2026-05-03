#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI extremes and choppiness regime filter.
# In trending markets (CHOP < 38.2), follow KAMA direction (long if price > KAMA, short if price < KAMA).
# In ranging markets (CHOP > 61.8), mean revert at RSI extremes (long RSI<30, short RSI>70).
# Uses 1w HTF for regime strength: only trade when 1w trend aligns with 1d signal.
# Discrete position sizing (0.25) to minimize fee churn. Target: 15-25 trades/year.

name = "1d_KAMA_RSI_Chop_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Indicators ===
    # KAMA ( Kaufman Adaptive Moving Average )
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1.0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (CHOP) - 14 period
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close_1d, 1))), np.abs(low - np.roll(close_1d, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_14[0] = np.mean(np.maximum(np.maximum(high[0] - low[0], np.abs(high[0] - close_1d[0])), np.abs(low[0] - close_1d[0])))
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if atr_14[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(atr_14[i-13:i+1]) / np.log(10) / (max_high[i] - min_low[i]))
        else:
            chop[i] = 50.0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 1w Trend Filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_1w_val = ema_20_1w_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(ema_1w_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        is_1w_uptrend = close_val > ema_1w_val
        is_1w_downtrend = close_val < ema_1w_val
        
        # Generate signals based on regime
        if position == 0:
            if is_trending and is_1w_uptrend and close_val > kama_val:
                # Long in uptrend, price above KAMA
                signals[i] = 0.25
                position = 1
            elif is_trending and is_1w_downtrend and close_val < kama_val:
                # Short in downtrend, price below KAMA
                signals[i] = -0.25
                position = -1
            elif is_ranging and rsi_val < 30:
                # Long at RSI oversold in ranging market
                signals[i] = 0.25
                position = 1
            elif is_ranging and rsi_val > 70:
                # Short at RSI overbought in ranging market
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend change or KAMA cross below
            if not is_trending or close_val < kama_val or not is_1w_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend change or KAMA cross above
            if not is_trending or close_val > kama_val or not is_1w_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals