#!/usr/bin/env python3
# 6H_RSI2_EDGE_1DTrend_Volume
# Hypothesis: Use a 2-period RSI (RSI2) with overbought/oversold thresholds on 6h timeframe, filtered by 1d trend (EMA50) and volume spike (2x average volume). RSI2 captures short-term momentum extremes that often reverse in ranging markets, while the 1d EMA50 filter ensures we only take trades in the direction of the higher timeframe trend. Volume spike confirms conviction. Designed for low trade frequency (15-25 trades/year) to minimize fee drag, effective in both bull and bear markets due to trend filter.

name = "6H_RSI2_EDGE_1DTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 2-period RSI on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Volume spike: 2x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # RSI2 and vol MA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi2[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI2 < 15 (oversold), price above daily EMA50 (uptrend), volume spike (>2x)
            if (rsi2[i] < 15 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 85 (overbought), price below daily EMA50 (downtrend), volume spike (>2x)
            elif (rsi2[i] > 85 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI2 returns to neutral (>45) or price closes below daily EMA50
            if (rsi2[i] > 45 or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI2 returns to neutral (<55) or price closes above daily EMA50
            if (rsi2[i] < 55 or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals