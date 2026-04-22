#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above upper BB (20,2) + ADX > 25 (trending) + volume > 1.5x average
# Short when price breaks below lower BB (20,2) + ADX > 25 (trending) + volume > 1.5x average
# Exit when price returns to middle BB (20-period SMA) or ADX < 20 (range)
# Bollinger Band squeeze identifies low volatility periods preceding explosive moves
# ADX filters for trending markets to avoid false breakouts in ranging conditions
# Designed for low trade frequency (~15-30/year) to minimize fee drain.
# Works in bull/bear by capturing breakouts in trending markets only.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d data for trend strength
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Bollinger Bands on 4h data (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 20-period average volume for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: break above upper BB + ADX > 25 (trending) + volume confirmation
            if price > upper_bb[i] and adx_val > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower BB + ADX > 25 (trending) + volume confirmation
            elif price < lower_bb[i] and adx_val > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: return to middle BB or ADX < 20 (range)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle BB or trend weakens
                if price < sma_20[i] or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle BB or trend weakens
                if price > sma_20[i] or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BB_Squeeze_ADX25_Volume"
timeframe = "4h"
leverage = 1.0