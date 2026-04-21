#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for key indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ADX (14-period) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_smooth = pd.Series(atr_14).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align daily ADX to any timeframe (we'll use it as filter)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1-hour data for entry timing
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # 1h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Align Bollinger Bands
    bb_upper_aligned = align_htf_to_ltf(prices, df_1h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1h, bb_lower)
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        # Volume filter: avoid low volume chop
        vol_filter = vol > 0.8 * vol_ma
        
        if position == 0:
            # Long: price touches lower BB in trending market with volume
            if price <= bb_lower_val and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB in trending market with volume
            elif price >= bb_upper_val and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses the middle Bollinger Band (SMA20)
            sma_20_val = sma_20[i] if i < len(sma_20) else sma_20[-1]
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses above SMA20
                if price > sma_20_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses below SMA20
                if price < sma_20_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1h_BollingerTouch_ADX25_VolumeFilter"
timeframe = "1h"
leverage = 1.0