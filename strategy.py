#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI(14) mean reversion + 1d volume spike filter
# Long when KAMA direction is up AND RSI < 30 (oversold) AND 1d volume > 1.5x 20-day average
# Short when KAMA direction is down AND RSI > 70 (overbought) AND 1d volume > 1.5x 20-day average
# Exit when RSI crosses back to neutral (40-60 range) OR KAMA direction flips
# KAMA adapts to market noise, reducing whipsaws in choppy markets
# RSI extremes provide mean reversion entries in both bull and bear markets
# Volume spike confirms institutional participation at turning points
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_KAMA_RSI_MeanReversion_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average for spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma_20_1d)
    
    # Align 1d volume spike to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate KAMA (adaptive moving average) on 12h close
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close - close[10]|
    volatility = np.abs(np.subtract(close[1:], close[:-1]))  # |close - close[1]|
    
    # Pad arrays for rolling sum
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    # Calculate 10-period rolling sums
    change_sum = pd.Series(change_padded).rolling(window=10, min_periods=10).sum().values[10:]
    volatility_sum = pd.Series(volatility_padded).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.divide(change_sum, volatility_sum, out=np.full_like(change_sum, 0.1), where=volatility_sum!=0)
    er = np.concatenate([np.full(10, 0.1), er])  # Pad beginning
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # Calculate RSI(14) on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle all gains case
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any value is NaN
        if (np.isnan(kama_dir[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up AND RSI < 30 (oversold) AND volume spike
            if (kama_dir[i] > 0 and 
                rsi[i] < 30 and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down AND RSI > 70 (overbought) AND volume spike
            elif (kama_dir[i] < 0 and 
                  rsi[i] > 70 and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back to neutral (>=40) OR KAMA direction flips down
            if (rsi[i] >= 40 or 
                kama_dir[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back to neutral (<=60) OR KAMA direction flips up
            if (rsi[i] <= 60 or 
                kama_dir[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals