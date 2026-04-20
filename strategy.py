#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi_1d = 100.0 - (100.0 / (1.0 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h ATR (14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h EMA (50) for trend filter
    close_6h = prices['close'].values
    ema_50 = pd.Series(close_6h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        ema_50_val = ema_50[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(atr_1d_val) or 
            np.isnan(atr_6h_val) or np.isnan(ema_50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with price above EMA50 and sufficient volatility
            if rsi_val < 30 and close_val > ema_50_val and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with price below EMA50 and sufficient volatility
            elif rsi_val > 70 and close_val < ema_50_val and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or ATR-based stop
            if rsi_val > 70 or close_val < prices['high'].iloc[i] - 2.0 * atr_6h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or ATR-based stop
            if rsi_val < 30 or close_val > prices['low'].iloc[i] + 2.0 * atr_6h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_1dRSI_EMA50_ATRFilter_V1
# Uses 1-day RSI for mean reversion signals with 6h EMA50 trend filter
# Enters long when 1d RSI < 30 (oversold) and 6h price above EMA50
# Enters short when 1d RSI > 70 (overbought) and 6h price below EMA50
# Uses 1d ATR as volatility filter to avoid choppy markets
# Exits on opposite RSI extreme or 2*ATR stoploss (using 6h ATR)
# Designed for 6h timeframe with ~12-37 trades/year
name = "6h_1dRSI_EMA50_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0