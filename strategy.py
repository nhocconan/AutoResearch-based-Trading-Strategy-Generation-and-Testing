#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with RSI and chop regime filter for mean reversion and trend following.
# Uses KAMA(10) for trend direction, RSI(14) for mean reversion entries, and Choppiness Index(14) for regime detection.
# Long when KAMA up + RSI < 40 + chop > 61.8 (range), short when KAMA down + RSI > 60 + chop > 61.8.
# Exit when RSI crosses 50 or chop < 38.2 (trend). Designed for 1d timeframe to work in both bull and bear markets.
# Target: 15-25 trades/year per symbol to minimize fee drag.
name = "1d_KAMA_RSI_Chop_MeanRev_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for trend filter (optional, but can add robustness)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA components
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_s.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        chop_range = chop[i] > 61.8
        chop_trend = chop[i] < 38.2
        
        if position == 0:
            # Long: KAMA up + RSI oversold + chop range (mean reversion in range)
            if kama_up and rsi_oversold and chop_range:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + chop range (mean reversion in range)
            elif kama_down and rsi_overbought and chop_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses 50 or chop trend (trend resumption)
            if rsi[i] >= 50 or chop_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses 50 or chop trend (trend resumption)
            if rsi[i] <= 50 or chop_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals