#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend direction, 4h Camarilla R1/S1 for entry, volume spike (1.5x avg) for confirmation.
# Designed for 1h timeframe: 1h for entry timing, 4h/1d for direction. Target: 15-30 trades/year.
# Works in bull/bear by following daily trend while entering on 4h Camarilla breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 4h data for Camarilla pivot and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Camarilla pivot levels on 4h (using previous 4h bar's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous bar's OHLC to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r1 = np.full(len(close_4h), np.nan)
    camarilla_s1 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):  # Start from 1 to use previous bar
        h = high_4h[i-1]
        l = low_4h[i-1]
        c = close_4h[i-1]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12
    
    # Calculate 20-period average volume on 4h for spike detection
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(vol_4h), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-vol_period:i])
    
    # Align all indicators to 1h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # Conservative size to manage drawdown
    
    # Warmup period: need enough data for all indicators
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 with uptrend and volume
            if price > camarilla_r1_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Price breaks below Camarilla S1 with downtrend and volume
            elif price < camarilla_s1_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Camarilla S1 or trend reversal
            if price < camarilla_s1_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Camarilla R1 or trend reversal
            if price > camarilla_r1_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1S1_1dEMA34_Volume_Session"
timeframe = "1h"
leverage = 1.0