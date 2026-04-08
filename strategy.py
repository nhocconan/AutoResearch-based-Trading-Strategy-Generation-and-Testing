#!/usr/bin/env python3
# 1d_kama_rsi_chop_filter_v1
# Hypothesis: Use daily KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
# Long when KAMA is rising, RSI > 50, and CHOP > 61.8 (ranging market) to capture mean reversion within an uptrend.
# Short when KAMA is falling, RSI < 50, and CHOP > 61.8 (ranging market) to capture mean reversion within a downtrend.
# Uses 1d timeframe with 1h trend filter to reduce false signals. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper KAMA calculation
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    ER = change / volatility.replace(0, np.nan)
    SC = (ER * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    SC = SC.fillna(0).values
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # Calculate RSI(14) on daily close
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate Choppiness Index on daily high/low/close
    # CHOP = 100 * log10(sum(ATR(14)) / (n * ATR(14))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (14 * atr)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # neutral when undefined
    
    # Get 1h data for trend filter (to avoid trading against higher timeframe trend)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1h close for trend filter
    close_1h = df_1h['close'].values
    ema50_1h = pd.Series(close_1h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # 1h uptrend/downtrend
    uptrend_1h = close > ema50_1h_aligned
    downtrend_1h = close < ema50_1h_aligned
    
    signals = np.zeros(n)
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or \
           np.isnan(ema50_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: KAMA rising, RSI > 50, choppy market (CHOP > 61.8), and 1h uptrend
        if kama_rising[i] and rsi[i] > 50 and chop[i] > 61.8 and uptrend_1h[i]:
            signals[i] = 0.25
        # Short entry: KAMA falling, RSI < 50, choppy market (CHOP > 61.8), and 1h downtrend
        elif kama_falling[i] and rsi[i] < 50 and chop[i] > 61.8 and downtrend_1h[i]:
            signals[i] = -0.25
        else:
            # No clear signal, stay flat
            signals[i] = 0.0
    
    return signals