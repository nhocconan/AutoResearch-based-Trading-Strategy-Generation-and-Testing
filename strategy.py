#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below R3/S3 pivots from daily timeframe (Camarilla) with daily trend filter (EMA34) and volume confirmation (1.5x 20-period average). Designed for low trade frequency (<30/year) with clear entries in both bull and bear markets by following daily trend direction. Uses 12h timeframe for entries to reduce frequency and improve signal quality.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for Camarilla pivots, EMA34 trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First day has no previous - set to current values
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, R3)
    s3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily trend filter: EMA34 on close
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume confirmation: 20-period average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(
        window=20, min_periods=20
    ).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma_12h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average
        volume_spike = volume[i] > (vol_ma_12h[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34 + volume spike
            if (close[i] > r3_12h[i] and 
                close[i] > ema_34_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume spike
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema_34_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to EMA34 or opposite pivot level
            if position == 1:
                # Exit long: price returns to EMA34 or breaks below S3
                if (close[i] <= ema_34_12h[i]) or \
                   (close[i] < s3_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA34 or breaks above R3
                if (close[i] >= ema_34_12h[i]) or \
                   (close[i] > r3_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals