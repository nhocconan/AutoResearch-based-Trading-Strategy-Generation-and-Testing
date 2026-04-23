#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA34 trend filter and volume confirmation.
Long when price crosses above Alligator Lips AND 1d EMA34 rising AND volume > 1.3x 20-period average.
Short when price crosses below Alligator Jaw AND 1d EMA34 falling AND volume > 1.3x 20-period average.
Exit when price re-enters Alligator's mouth (between Jaw and Lips) or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Williams Alligator identifies trend absence (sleeping) vs trend awakening; 1d EMA34 filters counter-trend moves.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Works in both bull and bear markets by trading with the 1d trend and using volume confirmation to filter false signals.
"""

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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 12h Williams Alligator (SMMA with specific periods)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Jaw (Blue) - 13-period SMMA, 8 bars ahead
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    
    # Teeth (Red) - 8-period SMMA, 5 bars ahead
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    
    # Lips (Green) - 5-period SMMA, 3 bars ahead
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_1d_34_aligned)
    ema_slope[1:] = ema_1d_34_aligned[1:] - ema_1d_34_aligned[:-1]
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 13, 8, 5, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_1d_34_aligned[i]) or np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaw_val = jaw_aligned[i]
        lips_val = lips_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price crosses above Lips AND 1d EMA34 rising AND volume spike
            if (price > lips_val and 
                ema_slope_val > 0 and 
                volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price crosses below Jaw AND 1d EMA34 falling AND volume spike
            elif (price < jaw_val and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price re-enters Alligator's mouth (between Jaw and Lips)
            if position == 1 and price < lips_val:
                exit_signal = True
            elif position == -1 and price > jaw_val:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0