#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h RSI for momentum, 1d volatility regime filter (ATR ratio), and volume confirmation.
# Uses tight entry conditions to limit trades (~20-30/year) and avoid overtrading.
# RSI > 55 for long, < 45 for short on 12h timeframe ensures momentum alignment.
# ATR ratio < 0.8 indicates low volatility regime for better breakout quality.
# Volume > 1.5x 20-period average confirms institutional participation.
name = "4h_12hRSI_1dATRratio_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # RSI(14) calculation
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Get 1d data for ATR (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # ATR(14) calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Current ATR for volatility regime (4h ATR)
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = high[0] - low[0]
    tr_4h2[0] = high[0] - close[0]
    tr_4h3[0] = low[0] - close[0]
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volatility regime filter: current ATR / 1d ATR < 0.8 (low vol regime)
    vol_regime = atr_4h / (atr_1d_aligned + 1e-10) < 0.8
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) + low vol regime + volume confirmation
            if (rsi_12h_aligned[i] > 55 and 
                vol_regime[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 (bearish momentum) + low vol regime + volume confirmation
            elif (rsi_12h_aligned[i] < 45 and 
                  vol_regime[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI drops below 50 (momentum fade) or volatility spikes
            if (rsi_12h_aligned[i] < 50 or 
                vol_regime[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI rises above 50 (momentum fade) or volatility spikes
            if (rsi_12h_aligned[i] > 50 or 
                vol_regime[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals