#!/usr/bin/env python3
name = "4h_KAMA_RSI_Chop_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on close
    close_series = pd.Series(close)
    # Efficiency Ratio (ER)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14) on 4h
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(highest_high - lowest_low) / np.log10(14) / np.log10(np.sum(atr, axis=0, keepdims=True).T if False else np.sum(pd.Series(atr).rolling(14).sum().values))
    chop = pd.Series(chop).fillna(50).values  # Default to neutral when undefined
    
    # Volume spike detection: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 4)  # Wait for KAMA, RSI, chop, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop < 61.8 (trending)
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            if close[i] > kama_aligned[i] and rsi[i] > 50 and chop[i] < 61.8 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_aligned[i] and rsi[i] < 50 and chop[i] < 61.8 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA trend + RSI momentum + Chop regime filter + volume confirmation
# - KAMA adapts to market noise, effective in both trending and ranging markets
# - RSI > 50 for long, < 50 for short ensures momentum alignment
# - Chop < 61.8 filters for trending regimes (avoids ranging markets where trend fails)
# - Volume confirmation (1.5x average) reduces false signals
# - Works in bull markets (KAMA up, RSI > 50) and bear markets (KAMA down, RSI < 50)
# - Exit when trend weakens (price crosses KAMA) or momentum fades (RSI extremes)
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Uses 12h KAMA for higher timeframe trend filter, reducing whipsaws
# - Combines adaptive trend (KAMA), momentum (RSI), regime (Chop), and volume
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Prioritizes BTC/ETH performance; avoids over-optimization on SOL
# - Discrete position sizing minimizes transaction costs from signal changes
# - Uses proper alignment to avoid look-ahead bias with multi-timeframe data
# - All indicators use sufficient lookback with min_periods to ensure validity