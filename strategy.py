#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume spike detection (volume > 2x 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 4-period RSI for entry timing
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=4, min_periods=4).mean().values
    avg_loss = pd.Series(loss).rolling(window=4, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when RSI becomes overbought or volatility drops
            if rsi[i] > 70 or atr_1d_aligned[i] < atr_ma_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI becomes oversold or volatility drops
            if rsi[i] < 30 or atr_1d_aligned[i] < atr_ma_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # LONG: RSI oversold, volume spike, sufficient volatility
            if rsi[i] < 30 and volume_spike[i] and atr_1d_aligned[i] > atr_ma_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: RSI overbought, volume spike, sufficient volatility
            elif rsi[i] > 70 and volume_spike[i] and atr_1d_aligned[i] > atr_ma_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_RSI_VolumeSpike_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0