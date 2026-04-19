# 1d_KAMA_Trend_Filter_RSI_Reversal
# Hypothesis: Daily KAMA trend direction filters RSI mean-reversion entries. Long when KAMA rising and RSI < 30, short when KAMA falling and RSI > 70. Exit when RSI crosses 50. Uses 1w trend filter to avoid counter-trend trades in strong trends. Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Expected trades: 15-25/year per symbol. Works by catching overextended moves in the direction of the higher timeframe trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_Filter_RSI_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly trend to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA on daily data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI on daily data (14-period)
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    # First average
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    # Subsequent averages
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Wait for weekly EMA50 and KAMA/RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or i < 1):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        kama_prev = kama[i-1]
        rsi_prev = rsi[i-1]
        
        if position == 0:
            # Long entry: KAMA rising (trend up), RSI oversold, and price above weekly EMA50
            if (kama_val > kama_prev and rsi_val < 30 and price > ema50):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling (trend down), RSI overbought, and price below weekly EMA50
            elif (kama_val < kama_prev and rsi_val > 70 and price < ema50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion complete)
            if rsi_prev <= 50 and rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion complete)
            if rsi_prev >= 50 and rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals