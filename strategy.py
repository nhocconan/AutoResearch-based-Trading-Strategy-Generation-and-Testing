#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter (EMA34) and volume confirmation.
Long when price breaks above Camarilla R3 AND 1w EMA34 rising AND volume > 2.0x 24-period MA.
Short when price breaks below Camarilla S3 AND 1w EMA34 falling AND volume > 2.0x 24-period MA.
Exit when price touches opposite Camarilla level (R4/S4) or 1w EMA34 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades in bear markets (2025+), volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla provides precise intraday levels, 1w EMA34 filters major trend, volume confirms breakout strength.
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
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # For 12h timeframe, we use daily OHLC to calculate levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    # Calculate daily OHLC from 12h bars (2 bars per day)
    # We need to group 12h bars into days
    daily_high = np.full(n, np.nan)
    daily_low = np.full(n, np.nan)
    daily_close = np.full(n, np.nan)
    
    for i in range(1, n):
        # Each day has 2x 12h bars
        if i % 2 == 1:  # Odd indices are second 12h bar of day
            daily_high[i] = max(high[i-1], high[i])
            daily_low[i] = min(low[i-1], low[i])
            daily_close[i] = close[i]  # Close of second bar
        else:  # Even indices are first 12h bar of day
            if i >= 2:
                daily_high[i] = max(high[i-2], high[i-1])
                daily_low[i] = min(low[i-2], low[i-1])
                daily_close[i] = close[i-1]  # Close of previous day's second bar
    
    # Forward fill daily values for use in current 12h bar
    for i in range(2, n):
        if np.isnan(daily_high[i]):
            daily_high[i] = daily_high[i-1]
        if np.isnan(daily_low[i]):
            daily_low[i] = daily_low[i-1]
        if np.isnan(daily_close[i]):
            daily_close[i] = daily_close[i-1]
    
    # Calculate Camarilla levels using previous day's OHLC
    for i in range(2, n):
        if (np.isnan(daily_high[i-1]) or np.isnan(daily_low[i-1]) or 
            np.isnan(daily_close[i-1])):
            continue
            
        high_prev = daily_high[i-1]
        low_prev = daily_low[i-1]
        close_prev = daily_close[i-1]
        
        pivot[i] = (high_prev + low_prev + close_prev) / 3
        range_prev = high_prev - low_prev
        
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 4
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 4
        camarilla_r4[i] = close_prev + range_prev * 1.1 / 2
        camarilla_s4[i] = close_prev - range_prev * 1.1 / 2
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h volume MA (24-period) for spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(2, 34, 24)  # Camarilla (needs 2), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_24[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 2.0x 24-period MA (higher threshold for fewer trades)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S4 (opposite extreme) OR EMA34 starts falling
                if price < s4 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R4 (opposite extreme) OR EMA34 starts rising
                if price > r4 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0