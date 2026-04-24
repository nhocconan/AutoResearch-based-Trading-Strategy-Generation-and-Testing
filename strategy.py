#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 (1d) AND 4h EMA50 rising AND volume > 1.5 * avg volume
- Short when price breaks below Camarilla S3 (1d) AND 4h EMA50 falling AND volume > 1.5 * avg volume
- Uses 1h for entry timing precision, 4h for trend direction, 1d for pivot levels
- Camarilla pivots calculated from prior 1d OHLC
- Volume filter avoids low-momentum breakouts
- Signal size: 0.20 discrete levels to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
- Session filter: 08-20 UTC to avoid low-volume Asian session
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots (use prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values  # shift(1) for prior completed bar
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla equations
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 1h timeframe (wait for 1d bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Trend: rising/falling EMA50
    ema_rising = ema_50_4h_aligned > np.roll(ema_50_4h_aligned, 1)
    ema_falling = ema_50_4h_aligned < np.roll(ema_50_4h_aligned, 1)
    # Handle first element
    ema_rising[0] = False
    ema_falling[0] = False
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 4h EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND EMA50 rising AND volume filter
            if close[i] > R3_aligned[i] and ema_rising[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND EMA50 falling AND volume filter
            elif close[i] < S3_aligned[i] and ema_falling[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR EMA50 turns flat/failing
            if close[i] < S3_aligned[i] or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 OR EMA50 turns flat/rising
            if close[i] > R3_aligned[i] or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0