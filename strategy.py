#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA50 rising AND volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 AND 1d EMA50 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (R2/S2) or 1d EMA50 reverses.
Uses 1d HTF for trend to avoid counter-trend trades and volume spike for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Camarilla levels provide adaptive support/resistance, 1d EMA50 filters major trend, volume confirms momentum.
Should work in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous bar's range (standard formula)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    
    for i in range(1, n):  # Need previous bar's data
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        if prev_range > 0:  # Avoid division by zero
            camarilla_R3[i] = prev_close + (prev_range * 1.1 / 4)
            camarilla_S3[i] = prev_close - (prev_range * 1.1 / 4)
            camarilla_R2[i] = prev_close + (prev_range * 1.1 / 6)
            camarilla_S2[i] = prev_close - (prev_range * 1.1 / 6)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla needs previous bar, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(camarilla_R2[i]) or np.isnan(camarilla_S2[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_R3[i]
        s3 = camarilla_S3[i]
        r2 = camarilla_R2[i]
        s2 = camarilla_S2[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA50 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA50 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S2 OR EMA50 starts falling
                if price < s2 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R2 OR EMA50 starts rising
                if price > r2 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0