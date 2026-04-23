#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA34 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND 4h EMA34 is falling AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for longs, R1 for shorts) or EMA34 reverses.
Uses 4h HTF for EMA34 trend (reduces whipsaws) and Camarilla levels from prior 4h bar.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate prior 4h bar's OHLC for Camarilla levels (using HTF data)
    # Camarilla levels based on prior 4h bar's range
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    camarilla_R1 = np.full(len(df_4h), np.nan)
    camarilla_S1 = np.full(len(df_4h), np.nan)
    camarilla_R4 = np.full(len(df_4h), np.nan)  # for stop loss reference
    camarilla_S4 = np.full(len(df_4h), np.nan)  # for stop loss reference
    
    for i in range(len(df_4h)):
        if i == 0:  # Need prior bar
            continue
        # Prior 4h bar (i-1)
        prior_high = high_4h[i-1]
        prior_low = low_4h[i-1]
        prior_close = close_4h[i-1]
        prior_range = prior_high - prior_low
        
        if prior_range <= 0:
            continue
            
        camarilla_R1[i] = prior_close + prior_range * 1.1 / 12
        camarilla_S1[i] = prior_close - prior_range * 1.1 / 12
        camarilla_R4[i] = prior_close + prior_range * 1.1 / 2
        camarilla_S4[i] = prior_close - prior_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S4)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        r4 = camarilla_R4_aligned[i]
        s4 = camarilla_S4_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA34 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 (opposite level) OR EMA34 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 (opposite level) OR EMA34 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0