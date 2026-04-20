#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_RSI_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # === 12h: KAMA for trend direction ===
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    
    # Efficiency Ratio
    change = np.abs(close_12h_s.diff(10))
    volatility = close_12h_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0)
    
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_12h[i] - kama[i-1])
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # === 4h: RSI for momentum ===
    close = prices['close'].values
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # === 4h: Volume confirmation ===
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        kama_val = kama_12h_aligned[i]
        rsi_val = rsi[i]
        current_close = close[i]
        current_volume = volume[i]
        
        if np.isnan(kama_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.3 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI > 55 (bullish momentum) + volume
            if current_close > kama_val and rsi_val > 55 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: price below KAMA (downtrend) + RSI < 45 (bearish momentum) + volume
            elif current_close < kama_val and rsi_val < 45 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 40
            if current_close <= kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 60
            if current_close >= kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals