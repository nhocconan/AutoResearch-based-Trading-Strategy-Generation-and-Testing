# Your Turn
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI filter and volatility regime detection.
# Uses KAMA to capture trend with low lag, RSI for mean reversion signals,
# and ATR-based volatility regime to filter trades in choppy markets.
# Designed for ~15-25 trades/year on 1d timeframe with strong trend signals
# that work in both bull and bear markets by avoiding choppy regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime filter (less noisy than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2))**2  # fast=2, slow=30
    
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full(n, np.nan)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volatility regime: ATR ratio (current vs 50-period average)
    atr_ma_50 = np.full(n, np.nan)
    for i in range(49, n):
        atr_ma_50[i] = np.mean(atr[i-49:i+1])
    
    vol_ratio = atr / atr_ma_50
    # Low volatility regime (trending): vol_ratio < 0.8
    # High volatility regime (choppy): vol_ratio > 1.2
    
    # Get weekly close for trend filter
    weekly_close = df_1w['close'].values
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        weekly_close_val = weekly_close_aligned[i]
        
        # Only trade in low volatility (trending) regime
        if vol_ratio_val >= 0.8:  # Avoid choppy markets
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA and RSI oversold (< 30)
            if price > kama_val and rsi_val < 30:
                signals[i] = size
                position = 1
            # Short: price below KAMA and RSI overbought (> 70)
            elif price < kama_val and rsi_val > 70:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_VolRegime"
timeframe = "1d"
leverage = 1.0