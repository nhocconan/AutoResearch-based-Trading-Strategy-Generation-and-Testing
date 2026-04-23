#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 4h EMA200 is rising AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 AND 4h EMA200 is falling AND volume > 1.8x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short) or reverses EMA200 direction.
Uses 4h HTF for EMA200 trend (avoids whipsaws in ranging markets) and 1d for regime filter (only trade when price > 1d EMA50 for longs, < 1d EMA50 for shorts).
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
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
    
    # Calculate 4h EMA200 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1d EMA50 for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC (1d lag)
        if i >= 24:  # Need at least 24 hours of 1h data for previous day
            prev_day_idx = i - 24
            if prev_day_idx >= 0 and prev_day_idx < n:
                # Get OHLC from 24 bars ago (previous day's close)
                ph = high[prev_day_idx:prev_day_idx+24].max()  # Previous day high
                pl = low[prev_day_idx:prev_day_idx+24].min()    # Previous day low
                pc = close[prev_day_idx:prev_day_idx+24].mean() # Previous day close
                
                # Camarilla levels
                camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4
                camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4
                camarilla_r4[i] = pc + (ph - pl) * 1.1 / 2
                camarilla_s4[i] = pc - (ph - pl) * 1.1 / 2
                camarilla_close[i] = pc
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 200, 50, 24)  # EMA200 (200), EMA50 (50), volume MA (20), Camarilla (24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_4h_val = ema_200_4h_aligned[i]
        ema_1d_val = ema_50_1d_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA200 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_4h_prev = ema_200_4h_aligned[i-1]
            ema_rising = ema_4h_val > ema_4h_prev
            ema_falling = ema_4h_val < ema_4h_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA200 rising AND price > 1d EMA50 AND volume spike
            if price > r3 and ema_rising and price > ema_1d_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND EMA200 falling AND price < 1d EMA50 AND volume spike
            elif price < s3 and ema_falling and price < ema_1d_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S3 OR EMA200 starts falling OR price < 1d EMA50
                if price < s3 or (i >= start_idx + 1 and ema_4h_val < ema_200_4h_aligned[i-1]) or price < ema_1d_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 OR EMA200 starts rising OR price > 1d EMA50
                if price > r3 or (i >= start_idx + 1 and ema_4h_val > ema_200_4h_aligned[i-1]) or price > ema_1d_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA200_Trend_1dEMA50_Regime_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0