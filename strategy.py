#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) with Elder Ray (Bull/Bear Power) filter and volume confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) AND Elder Ray Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator is bearish (Lips < Teeth < Jaw) AND Elder Ray Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator reverses (Lips crosses Teeth) or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 1d timeframe to target 10-25 trades/year per symbol (40-100 total over 4 years).
Works in both bull and bear markets by trading with the Alligator trend and using Elder Ray to filter weak moves.
1w Alligator provides multi-week trend structure; Elder Ray confirms bull/bear power behind the move.
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
    
    # Precompute session hours (00-23 UTC) - 1d timeframe trades all day
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 1w Williams Alligator (SMMA of median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    median_1w = (df_1w['high'] + df_1w['low']) / 2.0
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_raw = pd.Series(median_1w).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_raw = pd.Series(median_1w).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_raw = pd.Series(median_1w).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray to 1d timeframe (already 1d, so direct use)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period) on 1d timeframe
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
    start_idx = max(100, 13, 8, 5, 13, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d timeframe: no session filter needed
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume spike
            if (lips_val > teeth_val and teeth_val > jaw_val and
                bull_power_val > 0 and
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Alligator bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND volume spike
            elif (lips_val < teeth_val and teeth_val < jaw_val and
                  bear_power_val < 0 and
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator reverses (Lips crosses Teeth)
            if position == 1 and lips_val <= teeth_val:
                exit_signal = True
            elif position == -1 and lips_val >= teeth_val:
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

name = "1D_WilliamsAlligator_1dElderRay_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0