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
    
    # Get daily data for indicators (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20) for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 20)  # EMA34, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_14_1d_aligned[i]
        vol_filter = atr_14_1d_aligned[i] > atr_ma
        
        # Volume spike filter: volume > 1.5x average
        vol_spike = vol_ratio[i] > 1.5
        
        # Entry conditions: breakout with trend and volume
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume spike + volatility
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume spike + volatility
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if close[i] < donchian_low_20_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if close[i] > donchian_high_20_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DailyDonchian20_EMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0