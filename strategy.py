#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly RSI(14) for momentum
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Momentum filter: RSI not extreme
        rsi_mid = (rsi_14_1w_aligned[i] > 30) and (rsi_14_1w_aligned[i] < 70)
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1w_aligned[i] > 0 and atr_14_1w_aligned[i] < np.median(atr_14_1w_aligned[:i+1]) * 3
        
        # Volume filter: above average volume
        vol_ma_14_1w = pd.Series(volume_1w).rolling(window=14, min_periods=14).mean().values
        vol_ma_14_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_14_1w)
        if np.isnan(vol_ma_14_1w_aligned[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_14_1w_aligned[i]
        
        # Long conditions: neutral momentum + volatility filter + volume spike
        long_condition = (rsi_mid and vol_filter and vol_spike)
        
        # Short conditions: neutral momentum + volatility filter + volume spike
        short_condition = (rsi_mid and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI extreme
        elif position == 1 and rsi_14_1w_aligned[i] >= 70:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_14_1w_aligned[i] <= 30:
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

name = "1d_WeeklyRSI14_VolumeFilter_Session"
timeframe = "1d"
leverage = 1.0