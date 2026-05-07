#!/usr/bin/env python3
name = "4h_KAMA_Trend_With_RSI_Momentum"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h KAMA for trend
    close_12h = df_12h['close']
    # Calculate ER and SC for KAMA
    change = abs(close_12h - close_12h.shift(10))
    volatility = abs(close_12h.diff()).rolling(window=10).sum()
    ER = change / volatility
    ER = ER.fillna(0)
    SC = (ER * (0.6667 - 0.0645) + 0.0645) ** 2
    KAMA = [0.0] * len(close_12h)
    if len(close_12h) > 0:
        KAMA[0] = close_12h.iloc[0]
        for i in range(1, len(close_12h)):
            KAMA[i] = KAMA[i-1] + SC[i] * (close_12h.iloc[i] - KAMA[i-1])
    kama_12h = np.array(KAMA)
    
    # 12h RSI for momentum
    delta = close_12h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50)
    
    # Align to 4h
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h.values)
    
    # 4h RSI for entry timing
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0)
    avg_gain_4h = gain_4h.rolling(window=14, min_periods=14).mean()
    avg_loss_4h = loss_4h.rolling(window=14, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.fillna(50).values
    
    # 4h volume spike: > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or np.isnan(rsi_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI12h > 50, RSI4h > 50, volume spike
            if close[i] > kama_12h_aligned[i] and rsi_12h_aligned[i] > 50 and rsi_4h[i] > 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI12h < 50, RSI4h < 50, volume spike
            elif close[i] < kama_12h_aligned[i] and rsi_12h_aligned[i] < 50 and rsi_4h[i] < 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or RSI12h < 40
            if close[i] < kama_12h_aligned[i] or rsi_12h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or RSI12h > 60
            if close[i] > kama_12h_aligned[i] or rsi_12h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA trend with 12h RSI momentum filter and 4h RSI timing.
# Long when price > 12h KAMA, 12h RSI > 50 (bullish momentum), 4h RSI > 50, and volume spike confirms.
# Short when price < 12h KAMA, 12h RSI < 50, 4h RSI < 50, and volume spike confirms.
# Uses 12h timeframe for trend/momentum to reduce whipsaws, 4h for entry timing.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# KAMA adapts to market noise, reducing false signals in ranging markets.
# Works in bull markets (trend + momentum) and bear markets (reverse criteria).
# Target: 20-50 trades/year to minimize fee drag while capturing sustained moves.