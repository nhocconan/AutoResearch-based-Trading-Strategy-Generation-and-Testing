#!/usr/bin/env python3
"""
6h_ado_v1
Hypothesis: Uses ADO (Average Directional Oscillator) on 6h with 12h trend filter to capture trend continuation and mean reversion in different regimes. Long when ADO > 0 and price above 12h EMA50, short when ADO < 0 and price below 12h EMA50. Includes volume confirmation to avoid false signals. Designed to work in trending markets (ADO captures trend strength) and ranging markets (mean reversion at extremes) by using 12h trend filter to align with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ado_v1"
timeframe = "6h"
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
    
    # Calculate ADO (Average Directional Oscillator) = (DI+ - DI-) / (DI+ + DI-) * 100
    # Using 14-period for DI calculation
    period = 14
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # Initialize with SMA of first 'period' values
            result[period-1] = np.nanmean(data[:period])
            # Apply Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = result[i-1]
        return result
    
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # ADX and ADO
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    ado = plus_di - minus_di  # This is the ADO (-100 to +100)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period*2, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(ado[i]) or np.isnan(tr_smoothed[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(vol_ma[i]) or tr_smoothed[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: ADO turns negative or trend changes
            if ado[i] < 0 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADO turns positive or trend changes
            if ado[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: ADO positive (bullish momentum) and price above 12h EMA50 (uptrend)
            if ado[i] > 0 and close[i] > ema_50_12h_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: ADO negative (bearish momentum) and price below 12h EMA50 (downtrend)
            elif ado[i] < 0 and close[i] < ema_50_12h_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals