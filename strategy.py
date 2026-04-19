#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA + daily RSI with weekly volatility filter for trend-following in bull markets and mean-reversion in bear markets.
# Uses KAMA to detect trend direction on 12h, RSI on daily for overbought/oversold conditions, and weekly ATR percentile to filter low-volatility regimes.
# Designed to work in both bull and bear markets by switching between trend-following (when volatility high) and mean-reversion (when volatility low).
# Target: 15-25 trades/year per symbol with disciplined entries to avoid fee drag.
name = "12h_KAMA_RSI_VolatilityFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily RSI(14) for overbought/oversold
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Weekly ATR percentile for volatility regime (low vol = mean reversion, high vol = trend following)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Percentile of current ATR over 52-week lookback (approx 1 year)
    atr_percentile = pd.Series(atr_14_1w).rolling(window=52, min_periods=20).rank(pct=True).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # 12-hour KAMA for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Efficiency Ratio
    change = abs(pd.Series(df_12h['close']).diff(10))
    volatility = pd.Series(df_12h['close']).diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(df_12h['close'], np.nan, dtype=float)
    kama[0] = df_12h['close'].iloc[0]
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (df_12h['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_values = kama
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: low volatility (<30th percentile) = mean reversion, high volatility (>70th) = trend following
        vol_regime = atr_percentile_aligned[i]
        
        if position == 0:
            # Mean reversion regime (low volatility): buy oversold, sell overbought
            if vol_regime < 0.3:
                if rsi_14_1d_aligned[i] < 30 and close[i] > kama_aligned[i]:  # Oversold but above KAMA (bullish bias)
                    signals[i] = 0.25
                    position = 1
                elif rsi_14_1d_aligned[i] > 70 and close[i] < kama_aligned[i]:  # Overbought but below KAMA (bearish bias)
                    signals[i] = -0.25
                    position = -1
            # Trend following regime (high volatility): follow KAMA direction
            elif vol_regime > 0.7:
                if close[i] > kama_aligned[i] and rsi_14_1d_aligned[i] > 50:  # Uptrend with bullish momentum
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_aligned[i] and rsi_14_1d_aligned[i] < 50:  # Downtrend with bearish momentum
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit: RSI overbought or price below KAMA
            if rsi_14_1d_aligned[i] > 70 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: RSI oversold or price above KAMA
            if rsi_14_1d_aligned[i] < 30 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals