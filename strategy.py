#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + RSI(14) mean reversion + volume spike filter.
# KAMA adapts to market noise, identifying true trend direction.
# RSI(14) < 30 or > 70 signals overextended conditions for mean reversion entries.
# Volume spike (>1.8x 20-period average) confirms conviction at reversal points.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "4h_KAMA_RSI_Volume_MeanReversion"
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
    
    # Get 4h data for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = pd.Series(df_4h['close'].values)
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = abs(close_4h - close_4h.shift(10))
    volatility = abs(close_4h - close_4h.shift(1)).rolling(window=10).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h.iloc[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_4h.iloc[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe (no additional delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama.values)
    
    # Calculate RSI(14) on 4h data
    delta = close_4h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when undefined
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi.values)
    
    # Calculate volume spike: current volume > 1.8 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long: Price below KAMA (dip in uptrend) AND RSI oversold AND volume spike
            if close_val < kama_val and rsi_val < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above KAMA (rally in downtrend) AND RSI overbought AND volume spike
            elif close_val > kama_val and rsi_val > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses above KAMA (trend resumption) OR RSI overbought
            if close_val > kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses below KAMA (trend resumption) OR RSI oversold
            if close_val < kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals