#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeSpike_ATRStop
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter, volume spike confirmation, and ATR-based stoploss. 
Designed for low trade frequency (7-25/year) on 1d timeframe to avoid fee drag while capturing strong trending moves in both bull and bear markets. 
Uses discrete position sizing (0.25) to minimize fee churn. Focus on BTC/ETH with SOL as secondary confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA and ATR
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # 1d EMA34 for ATR calculation (using close)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (more conservative than R3/S3)
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR on 1d data, aligned to 1d)
    # Calculate true range on 1d data
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (34), 1w EMA (50), volume MA (20), ATR (14)
    start_idx = max(34, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and uptrend (weekly EMA50)
            long_signal = (high_val > R1_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1w_val)
            # Short: price breaks below S1 with volume confirmation and downtrend (weekly EMA50)
            short_signal = (low_val < S1_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend reversal (weekly EMA50)
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend reversal (weekly EMA50)
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0