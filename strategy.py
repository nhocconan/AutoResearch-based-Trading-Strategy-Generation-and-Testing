#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend filter + 1d RSI mean reversion + volume spike
# KAMA adapts to market noise - slow in ranging markets, fast in trends
# 1d RSI < 30 for long, > 70 for short with volume confirmation (>2x average)
# Designed for low trade frequency (target: 30-60 total trades over 4 years)
# Works in bull markets (captures trend continuation) and bear markets (mean reversion from extremes)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[0:13] = np.nan  # First 13 values invalid
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA(10, 2, 30) - fast, slow, and length parameters
    er = np.abs(np.diff(close, prepend=close[0])) / (
        np.abs(np.diff(close, prepend=close[0])).rolling(window=10, min_periods=1).sum()
    )
    er[0] = 0
    sc = (er * (2/2 - 30/30) + 30/30) ** 2  # smoothing constant
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above KAMA (uptrend), RSI < 30 (oversold), volume spike
        if (close[i] > kama[i] and 
            rsi_1d_aligned[i] < 30 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below KAMA (downtrend), RSI > 70 (overbought), volume spike
        elif (close[i] < kama[i] and 
              rsi_1d_aligned[i] > 70 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI normalization
        elif position == 1 and (close[i] <= kama[i] or rsi_1d_aligned[i] > 50):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= kama[i] or rsi_1d_aligned[i] < 50):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_1dRSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0