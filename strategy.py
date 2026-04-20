#!/usr/bin/env python3
# 1d_1w_1wKAMA_RSI_Filter
# Hypothesis: Weekly KAMA trend on 1w timeframe with daily RSI mean reversion for entries.
# Uses 1w KAMA to capture long-term trend (works in bull/bear via trend filter) and daily RSI(14) < 30 or > 70 for mean-reversion entries.
# Target: 15-30 trades per year per symbol to minimize fee drag, works in all regimes via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_1wKAMA_RSI_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate weekly KAMA for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close_1w_series.diff(10))
    volatility = close_1w_series.diff().abs().rolling(window=10).sum()
    ER = change / volatility.replace(0, np.nan)
    # Smoothing constants
    SC = (ER * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan, dtype=float)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(SC.iloc[i]):
            kama_1w[i] = kama_1w[i-1] + SC.iloc[i] * (close_1w[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    
    # === Daily RSI(14) for mean reversion entries ===
    close = prices['close'].values
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    RS = avg_gain / avg_loss.replace(0, np.nan)
    RSI = 100 - (100 / (1 + RS))
    
    # Align weekly KAMA to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Get values
        close_val = close[i]
        kama_1w_val = kama_1w_aligned[i]
        rsi_val = RSI.iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_1w_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly KAMA (uptrend) and RSI oversold (<30)
            if close_val > kama_1w_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly KAMA (downtrend) and RSI overbought (>70)
            elif close_val < kama_1w_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or price crosses below KAMA
            if rsi_val > 50 or close_val < kama_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or price crosses above KAMA
            if rsi_val < 50 or close_val > kama_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals