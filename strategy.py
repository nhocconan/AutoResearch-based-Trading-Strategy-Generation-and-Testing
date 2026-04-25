#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Daily KAMA direction + RSI(14) extreme + Choppiness Index regime filter.
KAMA adapts to market noise, reducing whipsaws. RSI>70 or <30 provides mean-reversion edge in ranging markets.
Chop>61.8 confirms ranging regime where mean reversion works. Targets 7-25 trades/year by requiring all three conditions.
Works in bull/bear: KAMA catches trends, RSI extremes fade in ranges, chop filter avoids trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for KAMA, RSI, Chop (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA(10, 2, 30) - adaptive moving average
    close_series = pd.Series(df_1d['close'].values)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = pd.Series(index=close_series.index, dtype=float)
    kama.iloc[0] = close_series.iloc[0]
    for i in range(1, len(close_series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index(14)
    atr = pd.Series(np.zeros(len(df_1d)), index=df_1d.index)
    for i in range(len(df_1d)):
        if i == 0:
            atr.iloc[i] = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            atr.iloc[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
    atr_sum = atr.rolling(window=14, min_periods=14).sum()
    max_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # Align all indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA calculation (10) and indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama_aligned[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend) AND RSI < 30 (oversold) AND chop > 61.8 (ranging)
            long_condition = (curr_close > curr_kama) and (rsi_aligned[i] < 30) and (chop_aligned[i] > 61.8)
            # Short: price < KAMA (downtrend) AND RSI > 70 (overbought) AND chop > 61.8 (ranging)
            short_condition = (curr_close < curr_kama) and (rsi_aligned[i] > 70) and (chop_aligned[i] > 61.8)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < KAMA OR RSI > 50 (mean reversion) OR chop < 38.2 (trending)
            if (curr_close < curr_kama) or (rsi_aligned[i] > 50) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA OR RSI < 50 (mean reversion) OR chop < 38.2 (trending)
            if (curr_close > curr_kama) or (rsi_aligned[i] < 50) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0