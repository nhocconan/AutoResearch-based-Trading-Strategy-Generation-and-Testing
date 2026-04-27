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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_12h_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > atr_ma
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = vol_current > (vol_avg * 1.5)
        
        # Calculate Camarilla pivot levels from daily data
        # Camarilla: H4 = C + ((H-L)*1.1/2), L4 = C - ((H-L)*1.1/2)
        # But we use simpler R3/S3 and R4/S4 as in the strategy
        daily_idx = i // 4  # 6h bars per day = 4
        if daily_idx >= len(df_1d):
            signals[i] = 0.0
            continue
        high_1d_i = high_1d[daily_idx]
        low_1d_i = low_1d[daily_idx]
        close_1d_i = close_1d[daily_idx]
        
        # Calculate Camarilla levels
        range_1d = high_1d_i - low_1d_i
        r4 = close_1d_i + (range_1d * 1.1 / 2) * 2  # R4
        s4 = close_1d_i - (range_1d * 1.1 / 2) * 2  # S4
        r3 = close_1d_i + (range_1d * 1.1 / 2) * 1.5  # R3
        s3 = close_1d_i - (range_1d * 1.1 / 2) * 1.5  # S3
        
        if position == 0:
            # Long: breakout above R3 with trend and filters
            if close[i] > r3 and close[i] > ema_trend and vol_filter and volume_filter:
                signals[i] = size
                position = 1
            # Short: breakdown below S3 with trend and filters
            elif close[i] < s3 and close[i] < ema_trend and vol_filter and volume_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S3 or trend reverses
            if close[i] < s3 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R3 or trend reverses
            if close[i] > r3 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_Volume_Filter"
timeframe = "6h"
leverage = 1.0