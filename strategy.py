#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Pullback_Trend_Follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Trend Filter: EMA200 (bullish if price > EMA200) ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: 1 if close > EMA200 (bullish), -1 if close < EMA200 (bearish)
    trend_1d = np.where(close_1d > ema200_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h RSI(14) for pullback entries ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / np.where(loss_ma > 0, loss_ma, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h Volume Spike Filter ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        trend_val = trend_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(vol_ratio_val) or np.isnan(trend_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish trend + RSI pullback (<30) + volume spike
            if trend_val == 1 and rsi_val < 30 and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: bearish trend + RSI bounce (>70) + volume spike
            elif trend_val == -1 and rsi_val > 70 and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: RSI > 50 (momentum fade) or trend flip
            if rsi_val > 50 or trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (momentum fade) or trend flip
            if rsi_val < 50 or trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals