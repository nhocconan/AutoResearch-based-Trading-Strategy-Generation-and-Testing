#!/usr/bin/env python3
name = "1d_Momentum_Reversal_With_Volume"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA 20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before enough data
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: RSI < 30 (oversold) + above weekly EMA20 + volume surge
            if (rsi[i] < 30 and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + below weekly EMA20 + volume surge
            elif (rsi[i] > 70 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone or trend reversal
            if position == 1:
                # Exit long: RSI > 50 OR price below weekly EMA20
                if (rsi[i] > 50) or (close[i] < ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 OR price above weekly EMA20
                if (rsi[i] < 50) or (close[i] > ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals