#!/usr/bin/env python3
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
    
    # Load 1h and 12h data ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1h) < 10 or len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 1h Supertrend (ATR=10, multiplier=3)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    upper_band = (high + low) / 2 + 3 * atr
    lower_band = (high + low) / 2 - 3 * atr
    supertrend = np.ones_like(close) * np.nan
    for i in range(10, len(close)):
        if close[i] <= upper_band[i-1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = lower_band[i]
    st_direction = np.where(close > supertrend, 1, -1)
    
    # Calculate 12h RSI(14)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    st_direction_aligned = align_htf_to_ltf(prices, df_1h, st_direction)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(st_direction_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend (1h) AND RSI < 30 (12h) AND volume spike
            if (st_direction_aligned[i] == 1 and 
                rsi_12h_aligned[i] < 30 and 
                volume[i] > 2.0 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend (-1h) AND RSI > 70 (12h) AND volume spike
            elif (st_direction_aligned[i] == -1 and 
                  rsi_12h_aligned[i] > 70 and 
                  volume[i] > 2.0 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Supertrend reversal OR RSI returns to neutral zone
            if position == 1:
                if (st_direction_aligned[i] == -1 or 
                    rsi_12h_aligned[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (st_direction_aligned[i] == 1 or 
                    rsi_12h_aligned[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Supertrend1h_RSI12h_VolumeSpike"
timeframe = "4h"
leverage = 1.0