#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 (1h) AND price > 4h EMA50 AND volume > 1.5x average.
# Short when price breaks below Camarilla S3 (1h) AND price < 4h EMA50 AND volume > 1.5x average.
# Exit when price returns to Camarilla pivot point (1h) OR trend reversal.
# Uses 1h for entry timing, 4h for signal direction (trend filter) to reduce noise.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Target: 60-150 total trades over 4 years (15-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 1h data for Camarilla calculation (using previous completed bar)
    df_1h = prices  # primary timeframe is 1h
    # Calculate Camarilla levels for each 1h bar using previous bar's OHLC
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla formula: 
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1/2
    # S3 = Pivot - (H - L) * 1.1/2
    pivot = (high_prev + low_prev + close_prev) / 3.0
    camarilla_r3 = pivot + (high_prev - low_prev) * 1.1 / 2.0
    camarilla_s3 = pivot - (high_prev - low_prev) * 1.1 / 2.0
    camarilla_pivot = pivot  # for exit
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current 1h volume > 1.5x 20-period average
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter_1h = volume > (1.5 * vol_ma_1h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND price > 4h EMA50 AND volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema50_4h_aligned[i] and volume_filter_1h[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 AND price < 4h EMA50 AND volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema50_4h_aligned[i] and volume_filter_1h[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot OR trend reversal (price < 4h EMA50)
            if close[i] <= camarilla_pivot[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot OR trend reversal (price > 4h EMA50)
            if close[i] >= camarilla_pivot[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals