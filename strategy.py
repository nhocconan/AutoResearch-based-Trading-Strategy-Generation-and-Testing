#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_Strength_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d KAMA trend direction ===
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, 1))
    change = np.insert(change, 0, 0)  # align length
    volatility = np.abs(np.diff(close_1d, 1))
    volatility = np.insert(volatility, 0, 0)
    
    # Sum over 10 periods
    change_sum = np.convolve(change, np.ones(10), 'same')
    volatility_sum = np.convolve(volatility, np.ones(10), 'same')
    volatility_sum[volatility_sum == 0] = 1e-10  # avoid division by zero
    er = change_sum / volatility_sum
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # === 1d ATR for volatility filter ===
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # === 4h RSI for entry timing ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Volume filter ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: price above/below KAMA with sufficient volatility
            if close[i] > kama_1d_aligned[i] and atr14_1d_aligned[i] > np.nanmedian(atr14_1d_aligned[max(0, i-50):i+1]):
                # Uptrend: look for pullback to enter long
                if rsi[i] < 40 and volume[i] > vol_ma20[i]:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < kama_1d_aligned[i] and atr14_1d_aligned[i] > np.nanmedian(atr14_1d_aligned[max(0, i-50):i+1]):
                # Downtrend: look for bounce to enter short
                if rsi[i] > 60 and volume[i] > vol_ma20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: trend reversal or overbought
            if close[i] < kama_1d_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or oversold
            if close[i] > kama_1d_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h strategy using 1d KAMA as trend filter and 4h RSI for entry timing.
# Enters long on pullbacks in uptrend (RSI<40) and short on bounces in downtrend (RSI>60).
# Uses volatility filter to avoid choppy markets. Designed to work in both bull
# (trend following on pullbacks) and bear (mean reversion in downtrend bounces)
# markets. Targets ~50-100 trades over 4 years to minimize fee drag. Uses discrete
# sizing (0.25) to reduce churn. Works on BTC/ETH via institutional trend following
# with mean reversion entries.