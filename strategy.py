#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with RSI mean reversion and chop regime filter for BTC/ETH.
- Long when KAMA upward AND RSI < 30 AND chop > 61.8 (range condition)
- Short when KAMA downward AND RSI > 70 AND chop > 61.8 (range condition)
- Exit when RSI crosses 50 (mean reversion complete) or chop < 38.2 (trending regime)
- Uses 1d primary with 1w HTF for regime context to target 30-100 trades over 4 years (7-25/year)
- KAMA adapts to market noise, RSI captures exhaustion in ranging markets, chop filter avoids trending regimes
- Designed to work in sideways markets where mean reversion prevails (common in 2025+ bear/ranging)
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate 1d KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    # KAMA(10,2,30) - fast=2, slow=30
    close_series = pd.Series(daily_close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0).values
    kama = np.zeros_like(daily_close)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Trend filter: bullish if price > KAMA, bearish if price < KAMA
    bullish_regime = close > kama_aligned
    bearish_regime = close < kama_aligned
    
    # Calculate 1d RSI(14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Chopiness Index(14)
    atr = pd.Series(abs(high - low)).rolling(14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(14, min_periods=14).max().values
    ll = pd.Series(low).rolling(14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (hh - ll + 1e-10)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime filter: chop > 61.8 = ranging (good for mean reversion)
    ranging_regime = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 14)  # Need KAMA(30) and RSI/Chop(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up AND RSI oversold AND ranging regime
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and ranging_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI overbought AND ranging regime
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and ranging_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses 50 OR chop < 38.2 (trending regime)
            if rsi_aligned[i] > 50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses 50 OR chop < 38.2 (trending regime)
            if rsi_aligned[i] < 50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0