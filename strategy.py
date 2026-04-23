#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA50 rising AND 1h volume > 1.5x 20-period MA.
Short when price breaks below Camarilla S1 AND 4h EMA50 falling AND 1h volume > 1.5x 20-period MA.
Exit when price touches opposite Camarilla level (S1 for long, R1 for short) or 4h EMA50 reverses.
Uses 4h HTF for trend to avoid counter-trend trades, volume for momentum confirmation.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Session filter: 08-20 UTC to avoid low-volume Asian session noise.
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
    
    # Calculate 1h Camarilla levels (based on previous bar's OHLC)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)  # for exit
    camarilla_s3 = np.full(n, np.nan)  # for exit
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_ = ph - pl
        
        camarilla_r1[i] = pc + range_ * 1.1 / 12
        camarilla_s1[i] = pc - range_ * 1.1 / 12
        camarilla_r3[i] = pc + range_ * 1.1 / 4
        camarilla_s3[i] = pc - range_ * 1.1 / 4
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla (needs 1), EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
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
        
        # Volume filter: 1h volume > 1.5x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA50 rising AND volume filter
            if price > r1 and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA50 falling AND volume filter
            elif price < s1 and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S3 (strong opposite) OR EMA50 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 (strong opposite) OR EMA50 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0