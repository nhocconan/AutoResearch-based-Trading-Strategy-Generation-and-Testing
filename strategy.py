#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility filter. 
# Uses RSI(14) for momentum, 4h EMA50 for trend direction, and 1d ATR ratio for volatility regime.
# Only takes long when 4h EMA50 uptrend + 1h RSI < 30 (oversold) + low volatility regime.
# Only takes short when 4h EMA50 downtrend + 1h RSI > 70 (overbought) + low volatility regime.
# Volatility filter avoids choppy markets where RSI reversals fail. 
# Designed for 1-2 trades per week to minimize fee drag (~80-100/year).
# Works in bull/bear by following 4h trend and fading 1h extremes only in low vol.

name = "1h_RSI_4hEMA50_1dATR_VolFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for ATR volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # === 1d ATR for volatility regime (ATR14/ATR50) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / np.where(atr50 > 0, atr50, np.nan)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 1h RSI(14) for momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = atr_ratio_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade in low volatility regime (ATR ratio < 0.8)
        vol_filter = vol_ratio_val < 0.8
        
        if position == 0:
            # Long: 4h uptrend + RSI oversold + low volatility
            if ema_val > 0 and rsi_val < 30 and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI overbought + low volatility
            elif ema_val < 0 and rsi_val > 70 and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if rsi_val > 70 or ema_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if rsi_val < 30 or ema_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals