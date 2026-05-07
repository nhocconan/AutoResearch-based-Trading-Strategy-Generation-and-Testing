#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    dir = np.abs(np.diff(close_1w, k=10))  # direction over 10 periods
    vol = np.sum(change.reshape(-1, 10), axis=1)  # volatility over 10 periods
    er = np.where(vol != 0, dir / vol, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # k=2, n=30
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Weekly RSI for momentum filter
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi1w = 100 - (100 / (1 + rs))
    rsi1w = rsi1w.fillna(50).values  # Neutral when undefined
    rsi1w_aligned = align_htf_to_ltf(prices, df_1w, rsi1w)
    
    # Daily volume spike: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and KAMA
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly KAMA, RSI > 55, volume spike
            if close[i] > kama_1w_aligned[i] and rsi1w_aligned[i] > 55 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly KAMA, RSI < 45, volume spike
            elif close[i] < kama_1w_aligned[i] and rsi1w_aligned[i] < 45 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below weekly KAMA or RSI < 40
            if close[i] < kama_1w_aligned[i] or rsi1w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above weekly KAMA or RSI > 60
            if close[i] > kama_1w_aligned[i] or rsi1w_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly KAMA trend filter with RSI momentum and daily volume spike confirmation.
# Long when price > weekly KAMA, RSI > 55 (bullish momentum), and volume spike confirms conviction.
# Short when price < weekly KAMA, RSI < 45 (bearish momentum), and volume spike confirms.
# Uses weekly timeframe for trend/momentum to avoid whipsaws, daily for execution.
# Volume spike (>2.0x average) ensures strong conviction. Discrete 0.25 position size limits risk.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: ~15-25 trades/year to minimize fee drag while capturing sustained moves.