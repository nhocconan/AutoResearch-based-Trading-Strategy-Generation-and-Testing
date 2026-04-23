#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA50 rising AND 15m volume > 1.5x 20-period MA.
Short when price breaks below Camarilla S3 AND 12h EMA50 falling AND 15m volume > 1.5x 20-period MA.
Exit when price touches opposite Camarilla level (R3/S3) or 12h EMA50 reverses.
Uses 12h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Camarilla levels provide intraday structure, 12h EMA50 filters major trend, volume confirmation avoids low-momentum breakouts.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For 4h timeframe, we need daily OHLC to compute Camarilla
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for exit
    camarilla_s4 = np.full(n, np.nan)  # for exit
    
    # Get daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    for day_idx in range(len(df_1d)):
        # Get previous day's OHLC (for today's levels)
        if day_idx == 0:
            # For first day, use same day's OHLC (no previous day)
            prev_high = df_1d['high'].iloc[day_idx]
            prev_low = df_1d['low'].iloc[day_idx]
            prev_close = df_1d['close'].iloc[day_idx]
        else:
            prev_high = df_1d['high'].iloc[day_idx-1]
            prev_low = df_1d['low'].iloc[day_idx-1]
            prev_close = df_1d['close'].iloc[day_idx-1]
        
        # Camarilla formulas
        range_val = prev_high - prev_low
        camarilla_r3_day = prev_close + range_val * 1.1 / 4
        camarilla_s3_day = prev_close - range_val * 1.1 / 4
        camarilla_r4_day = prev_close + range_val * 1.1 / 2
        camarilla_s4_day = prev_close - range_val * 1.1 / 2
        
        # Find corresponding 4h bars for this day
        day_start_ts = df_1d.index[day_idx]
        if day_idx + 1 < len(df_1d):
            day_end_ts = df_1d.index[day_idx + 1]
        else:
            # For last day, extend to end of data
            day_end_ts = prices['open_time'].iloc[-1] + pd.Timedelta(days=1)
        
        # Mark 4h bars within this day
        day_mask = (prices['open_time'] >= day_start_ts) & (prices['open_time'] < day_end_ts)
        camarilla_r3[day_mask] = camarilla_r3_day
        camarilla_s3[day_mask] = camarilla_s3_day
        camarilla_r4[day_mask] = camarilla_r4_day
        camarilla_s4[day_mask] = camarilla_s4_day
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 15m volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA, EMA50
    
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
        
        # Volume filter: 15m volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
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
                # Long exit: price touches Camarilla S3 OR S4 OR EMA50 starts falling
                if price < s3 or price < s4 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 OR R4 OR EMA50 starts rising
                if price > r3 or price > r4 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0