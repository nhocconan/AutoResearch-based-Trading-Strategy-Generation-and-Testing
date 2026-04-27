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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    # Calculate 6h ATR(14) for position sizing (adjust based on volatility)
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] < np.median(atr_14_1d_aligned[:i+1]) * 3
        
        # Volume filter: above average volume (using 6h volume)
        vol_ma_14_6h = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
        if np.isnan(vol_ma_14_6h[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_14_6h[i]
        
        # Long conditions: bullish trend + volatility filter + volume spike
        long_condition = (price_above_ema and vol_filter and vol_spike)
        
        # Short conditions: bearish trend + volatility filter + volume spike
        short_condition = (price_below_ema and vol_filter and vol_spike)
        
        # Dynamic position sizing based on volatility (inverse volatility scaling)
        # Higher volatility = smaller position
        vol_ratio = atr_14_6h[i] / np.median(atr_14_6h[:i+1]) if np.median(atr_14_6h[:i+1]) > 0 else 1.0
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)  # Limit the range
        base_size = 0.25
        position_size = base_size / vol_ratio  # Inverse scaling
        position_size = np.clip(position_size, 0.15, 0.35)  # Keep within reasonable bounds
        
        if long_condition and position <= 0:
            signals[i] = position_size
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -position_size
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DailyEMA34_VolumeFilter_Session_DynamicSize"
timeframe = "6h"
leverage = 1.0