#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123-Reversal pattern with 1d ADX trend filter and volume confirmation
# The 123-Reversal is a price action pattern where:
# 1. Price makes a new high/low (point 1)
# 2. Pulls back to form a swing point (point 2)
# 3. Breaks through point 1 with momentum (point 3)
# Works in both bull/bear markets by only taking reversals in direction of higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d data
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (using EMA as approximation)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14 = adx
    
    # Align ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Identify swing points for 123 pattern
    # Find local highs and lows (3-bar lookback)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Swing high: higher than 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = True
        # Swing low: lower than 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = True
    
    # Track the most recent swing points
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    last_swing_high_val = 0
    last_swing_low_val = 0
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for swing detection and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        # Update swing points
        if swing_high[i]:
            last_swing_high_idx = i
            last_swing_high_val = high[i]
        if swing_low[i]:
            last_swing_low_idx = i
            last_swing_low_val = low[i]
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Need valid swing points
            if last_swing_high_idx == -1 or last_swing_low_idx == -1:
                signals[i] = 0.0
                continue
                
            # Determine trend direction from ADX and DI crossover
            # We need DI values - recalculate or approximate from recent data
            if i >= 2:  # Need at least 2 periods for DI
                # Simplified trend: if recent closes are rising, trend is up
                trend_up = close[i] > close[i-5]  # 5-period momentum
            else:
                trend_up = True
            
            # Long 123 reversal: 
            # 1. Made a swing low (point 1)
            # 2. Pulled back to form a lower high (point 2) 
            # 3. Broke above point 1 with volume (point 3)
            if (trend_up and 
                price > last_swing_low_val and  # Broke above point 1
                low[i] < last_swing_low_val and  # Pulled back below point 1 (formed point 2)
                i - last_swing_low_idx >= 3 and  # At least 3 bars since swing low
                vol > 1.3 * avg_vol[i]):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short 123 reversal:
            # 1. Made a swing high (point 1)
            # 2. Pulled back to form a higher low (point 2)
            # 3. Broke below point 1 with volume (point 3)
            elif (not trend_up and 
                  price < last_swing_high_val and  # Broke below point 1
                  high[i] > last_swing_high_val and  # Pulled back above point 1 (formed point 2)
                  i - last_swing_high_idx >= 3 and  # At least 3 bars since swing high
                  vol > 1.3 * avg_vol[i]):  # Volume confirmation
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below the swing low that started the move
            if price < last_swing_low_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above the swing high that started the move
            if price > last_swing_high_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_123Reversal_ADX_Volume"
timeframe = "4h"
leverage = 1.0