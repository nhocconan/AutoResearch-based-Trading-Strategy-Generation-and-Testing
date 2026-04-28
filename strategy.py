#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily 144-period EMA for trend filter (longer term trend)
    ema144_1d = pd.Series(close_1d).ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 6h timeframe
    ema144_aligned = align_htf_to_ltf(prices, df_1d, ema144_1d)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema144_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA144
        uptrend = close[i] > ema144_aligned[i]
        downtrend = close[i] < ema144_aligned[i]
        
        # Volatility filter: current ATR below 2x average (avoid extremely volatile markets)
        vol_filter = atr14_aligned[i] < (np.nanmean(atr14_aligned[max(0, i-50):i+1]) * 2.0)
        
        # Entry conditions: Donchian breakout with volume and trend
        long_breakout = close[i] > high_20[i]
        short_breakout = close[i] < low_20[i]
        
        long_entry = long_breakout and uptrend and vol_filter and (vol_ratio[i] > 1.5)
        short_entry = short_breakout and downtrend and vol_filter and (vol_ratio[i] > 1.5)
        
        # Exit conditions: return to opposite Donchian level or trend reversal
        long_exit = close[i] < low_20[i] or not uptrend
        short_exit = close[i] > high_20[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_1dEMA144_VolumeFilter"
timeframe = "6h"
leverage = 1.0