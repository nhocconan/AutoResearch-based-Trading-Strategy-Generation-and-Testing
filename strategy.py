#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility filter.
# Long when: 1h RSI(14) > 55, price > 4h EMA(50), and 1d ATR ratio < 0.8 (low volatility)
# Short when: 1h RSI(14) < 45, price < 4h EMA(50), and 1d ATR ratio < 0.8
# Exit when RSI crosses back to 50 or volatility increases.
# Uses momentum for entry, higher timeframe for trend direction, and volatility filter to avoid chop.
# Designed for ~20-30 trades/year per symbol.
name = "1h_RSI_EMA50_ATRFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 10-period SMA of ATR for ratio
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1d / (atr_ma_10 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_4h = ema_4h_50_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI > 55, price above 4h EMA50, low volatility (ATR ratio < 0.8)
            if rsi_val > 55 and price > ema_4h and atr_ratio_val < 0.8:
                signals[i] = 0.20
                position = 1
            # Short: RSI < 45, price below 4h EMA50, low volatility (ATR ratio < 0.8)
            elif rsi_val < 45 and price < ema_4h and atr_ratio_val < 0.8:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI < 50 or volatility increases (ATR ratio > 1.2)
            if rsi_val < 50 or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI > 50 or volatility increases (ATR ratio > 1.2)
            if rsi_val > 50 or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals