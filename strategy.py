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
    
    # Get daily data for trend context and entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily SMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    sma_50_1d = close_1d_series.rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily SMA(20) of ATR for volatility threshold
    atr_series = pd.Series(atr_1d)
    atr_sma_20_1d = atr_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily volume SMA(20) for volume filter
    volume_1d_series = pd.Series(volume_1d)
    vol_sma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h timeframe
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_sma_20_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily SMA50
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # Volatility filter: current ATR > 1.2 * its 20-period SMA (avoid low volatility)
        vol_filter = atr_1d_aligned[i] > 1.2 * atr_sma_20_1d_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period daily volume SMA
        vol_size_filter = volume[i] > 1.3 * vol_sma_20_1d_aligned[i]
        
        # Entry conditions: volatility + volume filters must pass
        long_entry = uptrend and vol_filter and vol_size_filter
        short_entry = downtrend and vol_filter and vol_size_filter
        
        # Exit conditions: trend reversal or volatility/volume drop
        long_exit = (not uptrend) or (not vol_filter) or (not vol_size_filter)
        short_exit = (not downtrend) or (not vol_filter) or (not vol_size_filter)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_sma50_atr_vol_filter_v1"
timeframe = "6h"
leverage = 1.0