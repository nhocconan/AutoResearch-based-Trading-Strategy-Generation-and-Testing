#!/usr/bin/env python3
# 12h_1d_w_rsi_divergence_volume_v1
# Strategy: 12h RSI divergence with volume confirmation and weekly trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: RSI divergences signal exhaustion in trends. Bullish divergence (price LL, RSI HL) in uptrend triggers long. Bearish divergence (price HH, RSI LH) in downtrend triggers short. Uses weekly EMA50 for trend filter and volume confirmation (>1.5x avg) to avoid false signals. Designed for low frequency (15-35/year) to minimize fee drag and work in both bull/bear regimes via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_w_rsi_divergence_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d RSI(14) for divergence detection
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i-1]) or 
            np.isnan(rsi_14_1d_aligned[i-2]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bull_div = (low[i] < low[i-1] < low[i-2]) and (rsi_14_1d_aligned[i] > rsi_14_1d_aligned[i-1] > rsi_14_1d_aligned[i-2])
        # Bearish divergence: price makes higher high, RSI makes lower high
        bear_div = (high[i] > high[i-1] > high[i-2]) and (rsi_14_1d_aligned[i] < rsi_14_1d_aligned[i-1] < rsi_14_1d_aligned[i-2])
        
        # Entry logic: RSI divergence + volume + trend alignment
        if bull_div and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_div and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or RSI extreme
        elif position == 1 and (not uptrend or rsi_14_1d_aligned[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or rsi_14_1d_aligned[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals