# 4h_1dATRBreakout_SpikeVolume_V1
# 4h strategy combining ATR-based breakout with volume spike and RSI filter
# ATR breakout (14) + volume spike (>1.5x SMA20) + RSI(14) > 50 for long, < 50 for short
# Exit when ATR-based stop is hit or RSI crosses opposite side
# Designed for 4h timeframe targeting 20-50 trades/year
# Works in both bull and bear markets via volatility-based breakouts and volume confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d average volume (20) for volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 4h RSI (14) for momentum filter
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        volume_val = prices['volume'].iloc[i]
        atr_val = atr_1d_aligned[i]
        avg_vol_val = avg_volume_1d_aligned[i]
        rsi_val = rsi[i]
        
        # Skip if any value is NaN
        if (np.isnan(atr_val) or np.isnan(avg_vol_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x average volume
        volume_spike = volume_val > 1.5 * avg_vol_val
        
        if position == 0:
            # Long: price breaks above close + ATR with volume spike and RSI > 50
            if close_val > close_val + atr_val and volume_spike and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below close - ATR with volume spike and RSI < 50
            elif close_val < close_val - atr_val and volume_spike and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below entry - ATR or RSI < 40
            if close_val < close_val - atr_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above entry + ATR or RSI > 60
            if close_val > close_val + atr_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_1dATRBreakout_SpikeVolume_V1
name = "4h_1dATRBreakout_SpikeVolume_V1"
timeframe = "4h"
leverage = 1.0