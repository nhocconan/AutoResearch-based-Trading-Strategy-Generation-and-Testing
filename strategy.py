#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_RSI_MeanReversion_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily KAMA Trend (1d) ===
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=9, prepend=close_1d[:9]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 9 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_trend = kama  # Rising if close > kama, falling if close < kama
    
    # === Daily RSI(14) for mean reversion ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA trend and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_trend)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold + volume confirmation
            if close_val > kama_val and rsi_val < 30 and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI overbought + volume confirmation
            elif close_val < kama_val and rsi_val > 70 and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or price below KAMA
            if rsi_val > 70 or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or price above KAMA
            if rsi_val < 30 or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals