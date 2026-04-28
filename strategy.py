#!/usr/bin/env python3
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
    
    # Get 12h data for trend filter and pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    open_12h = df_12h['open'].values
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla pivot levels (H4/L4 for entries, H3/L3 for exits)
    pivot = (high_12h + low_12h + close_12h) / 3
    range_hl = high_12h - low_12h
    H4 = close_12h + (range_hl * 1.1 / 2)
    L4 = close_12h - (range_hl * 1.1 / 2)
    H3 = close_12h + (range_hl * 1.1 / 4)
    L3 = close_12h - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 6h
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    
    # Get 6h data for volume and volatility
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    close_6h = df_6h['close'].values
    
    # Volume ratio (current 6h volume / 20-period average)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # ATR(14) for volatility filter
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    tr1 = np.abs(high_6h[1:] - low_6h[1:])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Volume filter: current 6h volume above average
        volume_filter = volume_6h[i] > vol_ma_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_6h_aligned[i] > 0.001 * close[i]  # At least 0.1% ATR
        
        # Entry conditions: Camarilla H4/L4 breakout with volume and trend
        long_breakout = close[i] > H4_aligned[i]
        short_breakout = close[i] < L4_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and vol_filter
        short_entry = downtrend and short_breakout and volume_filter and vol_filter
        
        # Exit conditions: Camarilla H3/L3 retracement
        long_exit = close[i] < H3_aligned[i]
        short_exit = close[i] > L3_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_VolumeTrend_12h"
timeframe = "6h"
leverage = 1.0