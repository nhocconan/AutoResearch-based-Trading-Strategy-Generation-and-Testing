#!/usr/bin/env python3
# 1h_RSI_MeanReversion_4hTrend_1dVolatility
# Hypothesis: In 1h timeframe, RSI extremes combined with 4h trend filter and 1d volatility regime
# provides mean-reversion entries during pullbacks in trending markets. Works in bull (buy dips in uptrend) 
# and bear (sell rallies in downtrend) with volatility filter to avoid ranging markets. Uses 4h for trend direction,
# 1d for volatility regime filter, and 1h only for precise entry timing via RSI.

name = "1h_RSI_MeanReversion_4hTrend_1dVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d data for volatility regime (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first tr is undefined
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    volatility_ratio = atr_1d / atr_ma_1d
    high_volatility = volatility_ratio > 1.2  # volatile regime
    
    # Align 1d volatility to 1h
    high_volatility_aligned = align_htf_to_ltf(prices, df_1d, high_volatility.astype(float))
    
    # 1h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(high_volatility_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in uptrend with high volatility
            if (rsi[i] < 30 and 
                trend_4h_up_aligned[i] > 0.5 and 
                high_volatility_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in downtrend with high volatility
            elif (rsi[i] > 70 and 
                  trend_4h_down_aligned[i] > 0.5 and 
                  high_volatility_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or trend fails
            if (rsi[i] > 70 or 
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI oversold or trend fails
            if (rsi[i] < 30 or 
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals