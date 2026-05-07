#!/usr/bin/env python3
name = "12h_KAMA_Trend_With_RSI_Momentum"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close']
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama_1d = np.array(kama)
    
    # 1d RSI
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    
    # Align 1d indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h volume spike: > 1.8x 8-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(8, 14)  # Wait for volume MA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 55, volume spike
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] > 55 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 45, volume spike
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] < 45 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or RSI < 40
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or RSI > 60
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h trend following with 1d KAMA trend filter and 1d RSI momentum filter.
# KAMA adapts to market noise, reducing false signals in ranging markets.
# Long when price > 1d KAMA, RSI > 55 (strong bullish momentum), and volume spike confirms conviction.
# Short when price < 1d KAMA, RSI < 45 (strong bearish momentum), and volume spike confirms.
# Uses 1d timeframe for trend/momentum to avoid whipsaws, 12h for entry timing to reduce frequency.
# Volume spike (>1.8x average) ensures institutional participation. Discrete 0.25 position size limits risk.
# Effective in both bull (trend + momentum) and bear (reverse criteria) markets.
# Target: 15-30 trades/year to minimize fee drag while capturing sustained moves.