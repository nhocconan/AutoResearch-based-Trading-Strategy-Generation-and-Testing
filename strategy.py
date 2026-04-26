#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) trend direction filtered by RSI extremes and Choppiness Index regime. Enter long when KAMA turns up, RSI < 30 (oversold), and market is choppy (CHOP > 61.8). Enter short when KAMA turns down, RSI > 70 (overbought), and market is choppy. Uses discrete position size 0.25. Designed for 7-25 trades/year on 1d by requiring confluence of trend, momentum, and regime filters, reducing overtrading while capturing mean-reversion in choppy markets and trend continuation in trending markets.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close (ER=10, fast=2, slow=30)
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / np.maximum(volatility, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Align 1w trend: EMA50 on 1w close
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) 14-period
    atr_series = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_series.shift(1)), abs(low - close_series.shift(1)))))
    tr = atr_series
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / np.maximum(hh - ll, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14), EMA50 (50)
    start_idx = max(10, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi.iloc[i]) or np.isnan(chop.iloc[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend: slope over 2 periods
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI extremes
        rsi_oversold = rsi.iloc[i] < 30
        rsi_overbought = rsi.iloc[i] > 70
        
        # Choppiness regime: choppy market
        choppy = chop.iloc[i] > 61.8
        
        # 1w trend filter: only trade in direction of 1w trend
        trend_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA up + RSI oversold + choppy + 1w uptrend
            long_signal = kama_up and rsi_oversold and choppy and trend_uptrend
            
            # Short: KAMA down + RSI overbought + choppy + 1w downtrend
            short_signal = kama_down and rsi_overbought and choppy and trend_downtrend
            
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
            # Exit: KAMA turns down OR RSI > 50 (exit oversold) OR choppy regime ends
            if (not kama_up or rsi.iloc[i] > 50 or chop.iloc[i] <= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI < 50 (exit overbought) OR choppy regime ends
            if (not kama_down or rsi.iloc[i] < 50 or chop.iloc[i] <= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0