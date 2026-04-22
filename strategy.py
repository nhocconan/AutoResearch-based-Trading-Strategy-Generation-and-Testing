#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean-reversion strategy using 4h RSI and 1d trend filter
# In bear markets (2025), RSI extremes on 4h with 1d downtrend filter provide high-probability short entries
# RSI < 30 on 4h indicates oversold conditions, but we only short when 1d trend is down (price < EMA50)
# Conversely, RSI > 70 on 4h with 1d uptrend (price > EMA50) for long entries
# Volume confirmation (>1.5x 20-period average) filters low-quality signals
# Designed for 1h timeframe targeting 15-30 trades/year with strict entry conditions to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for RSI calculation (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h[:13] = np.nan  # Not enough data for first 13 periods
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) on 4h + 1d uptrend + volume confirmation
            if (rsi_4h_aligned[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) on 4h + 1d downtrend + volume confirmation
            elif (rsi_4h_aligned[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                # Exit long: RSI > 50 or trend turns down
                if (rsi_4h_aligned[i] > 50 or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI < 50 or trend turns up
                if (rsi_4h_aligned[i] < 50 or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI40_1dEMA50_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0