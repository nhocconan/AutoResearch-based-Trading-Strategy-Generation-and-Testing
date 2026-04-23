#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 level AND 4h EMA50 rising AND 1h volume > 1.8x 20-period MA.
Short when price breaks below Camarilla S3 level AND 4h EMA50 falling AND 1h volume > 1.8x 20-period MA.
Exit when price touches opposite Camarilla level (R3/S3) or 4h EMA50 reverses.
Uses 4h HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Camarilla pivots provide intraday structure, 4h EMA50 filters major trend, volume avoids low-momentum false breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate 1h Camarilla pivots (based on previous day's OHLC)
    # For intraday, we use rolling window of 24 periods (24h = 1 day) to approximate daily OHLC
    lookback = 24  # 24 * 1h = 1 day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Previous day's OHLC (24 bars ago to 1 bar ago)
        prev_high = np.max(high[i-lookback:i])
        prev_low = np.min(low[i-lookback:i])
        prev_close = close[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_r3[i] = prev_close + range_val * 1.1 / 2
        camarilla_s3[i] = prev_close - range_val * 1.1 / 2
        camarilla_r4[i] = prev_close + range_val * 1.1
        camarilla_s4[i] = prev_close - range_val * 1.1
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Camarilla, EMA50, volume MA
    
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
        
        # Volume filter: 1h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA50 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND EMA50 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S3 OR EMA50 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 OR EMA50 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0