#!/usr/bin/env python3
"""
1h ATR Breakout with 4h Trend and Volume Confirmation
Hypothesis: ATR breakouts capture strong momentum moves. Filtered by 4h ADX trend to avoid chop and volume to avoid false signals. Works in bull/bear by aligning with 4h trend direction. Targets 15-35 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_atr_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h ADX(14) for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1h ATR(10) for breakout
    tr1h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1h[0] = np.nan
    atr = pd.Series(tr1h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1h EMA(20) for dynamic breakout level
    ema = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below EMA(20) OR ADX drops below 20 (trend weakening)
            if (close[i] <= ema[i] or adx_4h_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: close above EMA(20) OR ADX drops below 20 (trend weakening)
            if (close[i] >= ema[i] or adx_4h_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: close > EMA + 1.5*ATR, ADX > 25, volume
            if (close[i] > ema[i] + 1.5 * atr[i] and 
                adx_4h_aligned[i] > 25 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: close < EMA - 1.5*ATR, ADX > 25, volume
            elif (close[i] < ema[i] - 1.5 * atr[i] and 
                  adx_4h_aligned[i] > 25 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals