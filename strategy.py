#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA trend + RSI(2) mean reversion with 1-week trend filter
# In bull markets: go long when KAMA rising, RSI(2) < 10 (oversold), and weekly trend up
# In bear markets: go short when KAMA falling, RSI(2) > 90 (overbought), and weekly trend down
# Uses contrarian entries within the trend for high-probability mean reversion
# Target: 30-100 trades over 4 years (7-25/year) with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(2) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to daily timeframe (no alignment needed as same TF)
    # But we still use the helper for consistency in interface
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily ATR for stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price and indicators
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising (trend up), RSI(2) oversold, weekly trend up
            if kama_val > kama_aligned[i-1] and rsi_val < 10 and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: KAMA falling (trend down), RSI(2) overbought, weekly trend down
            elif kama_val < kama_aligned[i-1] and rsi_val > 90 and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (1.5x ATR below entry) or RSI(2) overbought
            if price <= entry_price - 1.5 * atr[i] or rsi_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (1.5x ATR above entry) or RSI(2) oversold
            if price >= entry_price + 1.5 * atr[i] or rsi_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI2_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0