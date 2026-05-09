#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily volatility filter and intraday momentum.
# Uses daily ATR-based volatility filter to avoid choppy markets and 4h RSI for mean reversion entries.
# Volatility filter ensures trades only occur during sufficient movement, reducing whipsaw in sideways markets.
# RSI(14) provides mean reversion signals with overbought/oversold thresholds.
# Target: 80-160 total trades over 4 years (20-40/year) with size 0.25.

name = "4h_ATR_Volatility_Filter_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation for daily data
    hl = df_1d['high'] - df_1d['low']
    hc = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    lc = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = hl[0]  # First value
    
    # ATR(14) using Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[0:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility regime filter)
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volatility filter: trade only when volatility is elevated (avoid chop)
    volatility_filter = atr_ratio_aligned > 0.8  # Only trade when volatility >= 80% of average
    
    # 4h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[0:14])
    avg_loss[13] = np.mean(loss[0:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI thresholds for mean reversion
    rsi_oversold = 30
    rsi_overbought = 70
    
    rsi_long_signal = rsi < rsi_oversold
    rsi_short_signal = rsi > rsi_overbought
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volatility_filter[i]) or 
            np.isnan(rsi_long_signal[i]) or np.isnan(rsi_short_signal[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + volatility filter
            if rsi_long_signal[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volatility filter
            elif rsi_short_signal[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or volatility drops
            if rsi[i] >= 50 or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or volatility drops
            if rsi[i] <= 50 or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals