#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend filter + 1d Bollinger Bands mean reversion + volume confirmation
# Long when ADX < 20 (ranging) AND price < BB lower AND volume > 1.5x avg
# Short when ADX < 20 (ranging) AND price > BB upper AND volume > 1.5x avg
# Exit when price returns to BB middle OR ADX > 25 (trending begins)
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear markets by fading extremes in ranging conditions

name = "12h_adx_bb_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX (14-period) from 12h - trend strength indicator
    # ADX < 20 = ranging market (good for mean reversion)
    # ADX > 25 = trending market (avoid mean reversion)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate([[high[0]], high[:-1]])) > 
                          (np.concatenate([[low[0]], low[:-1]]) - low), 
                          np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > 
                           (high - np.concatenate([[high[0]], high[:-1]])), 
                           np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean()
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
        
        return adx.values
    
    adx = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands (20, 2) from 1d timeframe for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate Bollinger Bands
    bb_period = 20
    bb_std = 2
    sma = pd.Series(daily_close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(daily_close).rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = sma + (std * bb_std)
    bb_lower = sma - (std * bb_std)
    bb_middle = sma
    
    # Align BB to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower.values)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(bb_middle_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to BB middle OR ADX > 25 (trending begins)
        if position == 1:  # long position
            if close[i] >= bb_middle_aligned[i] or adx[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] <= bb_middle_aligned[i] or adx[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (ADX < 20) with BB extremes
            # Long: price below BB lower in ranging market + volume confirmation
            if (adx[i] < 20 and close[i] < bb_lower_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above BB upper in ranging market + volume confirmation
            elif (adx[i] < 20 and close[i] > bb_upper_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals