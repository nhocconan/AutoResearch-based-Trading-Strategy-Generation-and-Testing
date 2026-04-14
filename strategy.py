#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d KAMA trend with volume confirmation and 1d RSI for mean reversion.
# Uses KAMA (Kaufman Adaptive Moving Average) to detect trend direction on 1d timeframe.
# Long when price is above KAMA, RSI < 40 (oversold), and volume > 1.5x average.
# Short when price is below KAMA, RSI > 60 (overbought), and volume > 1.5x average.
# Exit when price crosses back across KAMA or RSI reaches opposite extreme.
# Designed to capture mean reversion within trending markets, working in both bull and bear cycles.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio: |change over 10 periods| / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1) if len(close_1d) > 1 else np.array([])
    # Pad arrays to match length
    if len(change) > 0 and len(volatility) > 0:
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    else:
        er = np.array([])
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) > 30:
        kama[30] = close_1d[30]  # seed
        for i in range(31, len(close_1d)):
            if not np.isnan(sc[i-30]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i-30] * (close_1d[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need KAMA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for mean reversion entries
            # Long: price above KAMA, RSI oversold (<40), volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_1d_aligned[i] < 40 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below KAMA, RSI overbought (>60), volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_1d_aligned[i] > 60 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI reaches overbought (>60)
            if (close[i] <= kama_aligned[i] or 
                rsi_1d_aligned[i] >= 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI reaches oversold (<40)
            if (close[i] >= kama_aligned[i] or 
                rsi_1d_aligned[i] <= 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_RSI_MeanReversion_Volume_v1"
timeframe = "12h"
leverage = 1.0