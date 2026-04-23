#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 1w EMA34 is rising AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S1 AND 1w EMA34 is falling AND volume > 1.8x 20-period average.
Exit when price touches opposite Camarilla level (S1 for long, R1 for short) or EMA34 reverses direction.
Uses 1w HTF for EMA34 trend (avoids whipsaws in ranging markets). Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for exit
    camarilla_s4 = np.full(n, np.nan)  # for exit
    
    for i in range(1, n):
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla calculations
        range_ = prev_high - prev_low
        camarilla_r1[i] = prev_close + range_ * 1.1 / 12
        camarilla_s1[i] = prev_close - range_ * 1.1 / 12
        camarilla_r4[i] = prev_close + range_ * 1.1 / 2
        camarilla_s4[i] = prev_close - range_ * 1.1 / 2
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
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
            if price > r1 and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 (or S4 for stronger exit) OR EMA34 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 (or R4 for stronger exit) OR EMA34 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0