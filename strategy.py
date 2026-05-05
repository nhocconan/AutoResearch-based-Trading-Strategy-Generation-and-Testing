#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and choppiness filter
# Long when KAMA trending up AND RSI < 40 AND choppiness > 61.8 (range market)
# Short when KAMA trending down AND RSI > 60 AND choppiness > 61.8 (range market)
# Exit when RSI crosses 50 (mean reversion complete)
# Uses 1d primary timeframe with 1w HTF for trend confirmation
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# KAMA adapts to market noise, RSI captures mean reversion in chop, choppiness filter ensures range conditions

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close for trend
    # KAMA requires Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Pad ER to match close length (first 9 values are NaN)
    er_padded = np.full(n, np.nan)
    er_padded[9:] = er
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er_padded * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.nan_to_num(sc, nan=slow_sc**2)  # Handle NaN from padding
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start KAMA at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (no alignment needed as it's already 1d)
    kama_aligned = kama  # Already calculated on 1d close
    
    # Calculate 1w EMA50 for trend confirmation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.mean(gain[1:15])  # indices 1 to 14
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder's smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1)) / (ATR(14) * 14)) / log10(14)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # Align with close
    
    atr1 = tr1  # ATR(1) is just TR
    atr1_sum = np.full(n, np.nan)
    for i in range(14, n):
        atr1_sum[i] = np.nansum(atr1[i-13:i+1])  # Sum of last 14 TR values
    
    # Calculate ATR(14) using Wilder's method
    atr14 = np.full(n, np.nan)
    atr14[14] = np.nanmean(atr1[1:15])  # First ATR(14)
    for i in range(15, n):
        atr14[i] = (atr14[i-1] * 13 + atr1[i]) / 14
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    mask = (atr14 > 0) & (atr1_sum > 0) & (i >= 14)
    chop[14:] = 100 * np.log10(atr1_sum[14:] / (atr14[14:] * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA trending up (close > KAMA) AND RSI < 40 AND chop > 61.8 (range)
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 40 and 
                chop[i] > 61.8 and
                close[i] > ema_50_1w_aligned[i]):  # 1w trend confirmation
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA trending down (close < KAMA) AND RSI > 60 AND chop > 61.8 (range)
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 60 and 
                  chop[i] > 61.8 and
                  close[i] < ema_50_1w_aligned[i]):  # 1w trend confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals