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
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(50) for additional trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/both EMAs for long, below/both for short
        price_above_emas = close[i] > ema_34_1d_aligned[i] and close[i] > ema_50_1d_aligned[i]
        price_below_emas = close[i] < ema_34_1d_aligned[i] and close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] < np.median(atr_14_1d_aligned[:i+1]) * 2.5
        
        # Volume filter: above average volume (using daily volume MA)
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
        if np.isnan(vol_ma_20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_20_1d_aligned[i]
        
        # Long conditions: bullish trend + volatility filter + volume spike
        long_condition = (price_above_emas and vol_filter and vol_spike)
        
        # Short conditions: bearish trend + volatility filter + volume spike
        short_condition = (price_below_emas and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: trend reversal (price crosses below/above EMA34)
        elif position == 1 and close[i] < ema_34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_DualEMA_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0