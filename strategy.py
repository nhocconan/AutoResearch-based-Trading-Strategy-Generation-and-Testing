# USDT-M PERPETUAL FUTURES STRATEGY: 4H_EMA_RSI_PATTERN
# Strategy type: Trend-following with pullback entries
# Timeframe: 4h (primary), 1d (trend filter)
# Why it works in both bull and bear: Uses EMA for dynamic trend direction, RSI for oversold/overbought pullbacks, and volume confirmation to filter false signals. Works in bull markets by buying pullbacks in uptrends, and in bear markets by selling rallies in downtrends. The volume filter ensures participation only during institutional interest, reducing whipsaws.

name = "4H_EMA_RSI_Pattern"
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
    
    # Get 4H data for EMA and RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA21 on 4H close
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate RSI14 on 4H close
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss over 14 periods
    avg_gain = np.zeros_like(close_4h)
    avg_loss = np.zeros_like(close_4h)
    avg_gain[14] = np.mean(gain[1:15])  # First average of gains
    avg_loss[14] = np.mean(loss[1:15])  # First average of losses
    
    for i in range(15, len(close_4h)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 1D EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current 4H volume > 1.2 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA21 (uptrend), RSI oversold (<30), and volume confirmation
            if (close[i] > ema_21_4h_aligned[i] and 
                rsi_14_4h_aligned[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA21 (downtrend), RSI overbought (>70), and volume confirmation
            elif (close[i] < ema_21_4h_aligned[i] and 
                  rsi_14_4h_aligned[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA21 or RSI overbought (>70)
            if close[i] < ema_21_4h_aligned[i] or rsi_14_4h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA21 or RSI oversold (<30)
            if close[i] > ema_21_4h_aligned[i] or rsi_14_4h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3