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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) for momentum/mean reversion
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # RSI thresholds: oversold < 30, overbought > 70
        rsi_oversold = rsi_14_1d_aligned[i] < 30
        rsi_overbought = rsi_14_1d_aligned[i] > 70
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] < np.median(atr_14_1d_aligned[:i+1]) * 3
        
        # Volume filter: above average volume
        vol_ma_14_1d = pd.Series(volume_1d).rolling(window=14, min_periods=14).mean().values
        vol_ma_14_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_14_1d)
        if np.isnan(vol_ma_14_1d_aligned[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_14_1d_aligned[i]
        
        # Long conditions: RSI oversold + volatility filter + volume spike
        long_condition = (rsi_oversold and vol_filter and vol_spike)
        
        # Short conditions: RSI overbought + volatility filter + volume spike
        short_condition = (rsi_overbought and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi_14_1d_aligned[i] > 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_14_1d_aligned[i] < 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DailyRSI14_VolumeFilter_Session"
timeframe = "6h"
leverage = 1.0