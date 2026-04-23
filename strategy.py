#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1w EMA50 rising AND 12h volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 AND 1w EMA50 falling AND 12h volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (S3 for long, R3 for short) or 1w EMA50 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla levels provide structure, 1w EMA50 filters major trend, volume spike avoids low-momentum breakouts.
Designed for lower frequency to minimize fee drag and work in both bull and bear markets.
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
    
    # Calculate 12h Camarilla levels (R3, S3) based on previous bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for stop loss
    camarilla_s4 = np.full(n, np.nan)  # for stop loss
    camarilla_r3_prev = np.full(n, np.nan)
    camarilla_s3_prev = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's high, low, close to calculate current bar's levels
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        range_val = phigh - plow
        
        camarilla_r3[i] = pclose + range_val * 1.1 / 4
        camarilla_s3[i] = pclose - range_val * 1.1 / 4
        camarilla_r4[i] = pclose + range_val * 1.1 / 2
        camarilla_s4[i] = pclose - range_val * 1.1 / 2
        camarilla_r3_prev[i] = camarilla_r3[i-1] if i > 1 else np.nan
        camarilla_s3_prev[i] = camarilla_s3[i-1] if i > 1 else np.nan
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla needs 1 bar, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        r3_prev = camarilla_r3_prev[i]
        s3_prev = camarilla_s3_prev[i]
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
        
        # Volume filter: 12h volume > 2.0x 20-period MA (adaptive to volatility)
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
                # Long exit: price touches S3 (or S4 for stop) OR EMA50 starts falling
                if price <= s3 or price <= s4 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 (or R4 for stop) OR EMA50 starts rising
                if price >= r3 or price >= r4 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0