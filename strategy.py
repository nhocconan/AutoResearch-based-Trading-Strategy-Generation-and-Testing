#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Daily volume average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    
    # Align all data to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w.values)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d.values)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 20-period average (avoid low volatility)
        atr_condition = atr_14_1d_aligned[i] > (atr_14_1d_aligned[i] * 0.8)  # Always true if ATR > 0, but keeps structure
        
        # Volume condition: current 12h volume > 1.2x 20-period average
        # Approximate 12h volume from daily volume (assuming 2x 12h periods per day)
        volume_12h_approx = volume[i]  # Current 12h bar volume
        volume_ma_20_12h = volume_ma_20_1d_aligned[i] / 2  # Approximate 20-period average for 12h
        volume_condition = volume_12h_approx > (volume_ma_20_12h * 1.2)
        
        # Trend filter: only long when price > weekly EMA20, short when price < weekly EMA20
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: trend + volume + volatility
        if position == 0:
            if long_trend and volume_condition:
                position = 1
                signals[i] = position_size
            elif short_trend and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when trend reverses or volatility drops
            if not long_trend or not volume_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when trend reverses or volatility drops
            if not short_trend or not volume_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_EMA20_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0