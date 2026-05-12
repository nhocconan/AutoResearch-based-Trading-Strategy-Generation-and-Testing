#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout of Camarilla R3/S3 levels on 4h timeframe with 1d EMA50 trend filter and volume confirmation.
# Camarilla levels act as strong support/resistance in ranging markets, while breakouts capture momentum.
# 1d EMA50 filters trend direction, volume confirms breakout strength.
# Designed for 20-40 trades/year to minimize fee drag and work in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Camarilla levels from previous day (using 1d data) ===
    # Camarilla levels calculated from previous day's OHLC
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    # We use the previous day's data to avoid look-ahead
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].iloc[:-1].values
    prev_high = df_1d['high'].iloc[:-1].values
    prev_low = df_1d['low'].iloc[:-1].values
    
    # Calculate Camarilla R3 and S3 for previous day
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe (each 1d bar corresponds to 16 * 4h bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable (enough for EMA50 and previous day data)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, uptrend, volume
            if close[i] > camarilla_r3_aligned[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, downtrend, volume
            elif close[i] < camarilla_s3_aligned[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below Camarilla S3 or trend reversal
            if close[i] < camarilla_s3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Camarilla R3 or trend reversal
            if close[i] > camarilla_r3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals