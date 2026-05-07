#!/usr/bin/env python3
"""
4h_RSI_Pullback_12h_Trend_Volume
Hypothesis: Use 4-hour RSI pullbacks in the direction of the 12-hour trend with volume confirmation.
In bull markets, buy dips; in bear markets, sell rallies. The 12-hour trend filter ensures we trade with higher timeframe momentum, reducing counter-trend trades.
Volume confirmation ensures institutional participation. Targets 20-50 trades/year to minimize fee drag.
"""
name = "4h_RSI_Pullback_12h_Trend_Volume"
timeframe = "4h"
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
    
    # Get 12H data for trend filter and RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate RSI(14) on 12H close
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else np.nan
        avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else np.nan
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_12h = rsi[:len(close_12h)]
    
    # Calculate EMA(50) on 12H close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4H timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current 4H volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 40 (pullback), price above EMA50 (uptrend), volume confirmation
            if (rsi_12h_aligned[i] < 40 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (pullback), price below EMA50 (downtrend), volume confirmation
            elif (rsi_12h_aligned[i] > 60 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or price crosses below EMA50
            if (rsi_12h_aligned[i] > 60 or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or price crosses above EMA50
            if (rsi_12h_aligned[i] < 40 or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals