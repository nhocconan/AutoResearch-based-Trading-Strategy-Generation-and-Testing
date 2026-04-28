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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Bollinger Bands (20, 2)
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    
    # Align daily indicators to 12h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR above 1.5x average (avoid choppy markets)
        vol_filter = atr14_aligned[i] > (np.nanmean(atr14_aligned[max(0, i-50):i+1]) * 1.5)
        
        # Bollinger Band squeeze detection: narrow bands indicate low volatility
        bb_width = (upper_bb_aligned[i] - lower_bb_aligned[i]) / sma20[np.searchsorted(df_1d.index, prices.index[i]) if i < len(prices) else -1] if np.searchsorted(df_1d.index, prices.index[i]) < len(df_1d) else 20
        bb_squeeze = bb_width < 0.02  # Less than 2% width indicates squeeze
        
        # Entry conditions: breakout from Bollinger Bands with volume and volatility
        long_breakout = close[i] > upper_bb_aligned[i]
        short_breakout = close[i] < lower_bb_aligned[i]
        
        long_entry = long_breakout and vol_filter and bb_squeeze
        short_entry = short_breakout and vol_filter and bb_squeeze
        
        # Exit conditions: return to middle of Bollinger Bands
        middle_bb = sma20[np.searchsorted(df_1d.index, prices.index[i]) if i < len(prices) else -1] if np.searchsorted(df_1d.index, prices.index[i]) < len(df_1d) else sma20[-1]
        middle_bb_aligned = align_htf_to_ltf(prices, df_1d, pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values)
        
        long_exit = close[i] < middle_bb_aligned[i]
        short_exit = close[i] > middle_bb_aligned[i]
        
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

name = "12h_BollingerSqueeze_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0