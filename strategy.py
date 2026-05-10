#!/usr/bin/env python3
# 1h_RSI_OverboughtOversold_4hTrend_1dVolatility
# Hypothesis: RSI mean reversion on 1h with 4h trend filter and 1d volatility filter.
# Long when RSI < 30 and 4h close > 4h EMA20 (uptrend) and 1d ATR < 20-period average (low vol).
# Short when RSI > 70 and 4h close < 4h EMA20 (downtrend) and 1d ATR < 20-period average.
# Works in bull via oversold bounces in uptrend, bear via overbought reversals in downtrend.
# Low volatility filter avoids whipsaws in choppy markets. Target: 20-40 trades/year.

name = "1h_RSI_OverboughtOversold_4hTrend_1dVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Relative Strength Index"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def atr(high, low, close, period=14):
    """Average True Range"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift())
    tr3 = abs(low_s - close_s.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1d ATR for volatility filter
    atr_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate 1h RSI
    close = prices['close'].values
    rsi_1h = rsi(close, 14)
    
    # Low volatility condition: current ATR < 20-period average ATR
    low_vol = atr_1d_aligned < atr_ma_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) + EMA20 (20) + ATR (14) + ATR MA (20)
    start_idx = max(14, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1h[i]) or 
            np.isnan(ema20_4h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) AND 4h uptrend (close > EMA20) AND low volatility
            if rsi_1h[i] < 30 and close[i] > ema20_4h_aligned[i] and low_vol[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) AND 4h downtrend (close < EMA20) AND low volatility
            elif rsi_1h[i] > 70 and close[i] < ema20_4h_aligned[i] and low_vol[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (>70) OR 4h trend breaks (close < EMA20)
            if rsi_1h[i] > 70 or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI oversold (<30) OR 4h trend breaks (close > EMA20)
            if rsi_1h[i] < 30 or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals