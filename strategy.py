#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI momentum + chop filter for regime detection.
# Works in bull/bear by using KAMA for adaptive trend, RSI for momentum strength,
# and Choppiness Index to avoid ranging markets. Target: 7-25 trades/year (30-100 total).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA, RSI, and Chop - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive trend)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Calculate 1d RSI (momentum)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Choppiness Index (regime filter)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(df_1d['high'].values, df_1d['low'].values))
    tr2 = np.abs(np.subtract(df_1d['high'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr3 = np.abs(np.subtract(df_1d['low'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Align all indicators to 1d timeframe (using previous day's values)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend), RSI > 50 (momentum), Chop < 61.8 (trending)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8 and
                df_1d['volume'].values[-1] > vol_avg_aligned[i] if len(df_1d) > 0 else False):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend), RSI < 50 (momentum), Chop < 61.8 (trending)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8 and
                  df_1d['volume'].values[-1] > vol_avg_aligned[i] if len(df_1d) > 0 else False):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse conditions
            if position == 1:
                if (close[i] <= kama_aligned[i] or 
                    rsi_aligned[i] < 40 or 
                    chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] >= kama_aligned[i] or 
                    rsi_aligned[i] > 60 or 
                    chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_KAMA_RSI_Chop_Trend_Momentum"
timeframe = "1d"
leverage = 1.0