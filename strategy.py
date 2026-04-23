#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA50 rising AND 1h volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S1 AND 4h EMA50 falling AND 1h volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (S1 for long, R1 for short) or 4h EMA50 reverses.
Uses 4h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Camarilla pivots provide intraday structure, 4h EMA50 filters major trend, volume spike avoids low-momentum breakouts.
Designed for 1h timeframe with session filter (08-20 UTC) to reduce noise and overtrading.
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
    
    # Calculate Camarilla pivots (based on previous day's OHLC)
    # We'll use rolling window of 24 periods (1 day of 1h data) to approximate daily OHLC
    lookback = 24
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        # Get window for previous day approximation
        window_high = np.max(high[i-lookback+1:i+1])
        window_low = np.min(low[i-lookback+1:i+1])
        window_close = close[i]  # Current close as proxy for daily close
        pivot = (window_high + window_low + window_close) / 3.0
        range_val = window_high - window_low
        camarilla_r1[i] = pivot + range_val * 1.1 / 12.0
        camarilla_s1[i] = pivot - range_val * 1.1 / 12.0
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50, 20)  # Camarilla, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or
            hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
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
        
        # Volume filter: 1h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA50 rising AND volume filter AND session
            if price > r1 and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA50 falling AND volume filter AND session
            elif price < s1 and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 OR EMA50 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 OR EMA50 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0