#!/usr/bin/env python3
# 12h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Spike
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance levels.
# Breakouts above R3 or below S3 with volume confirmation and daily trend filter
# capture significant moves. Works in both bull and bear markets by following
# the daily trend direction. Designed for low trade frequency (~15-30/year) on 12h.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === Daily Data for Pivot Calculation and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Handle first value (no previous day)
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    camarilla_range = (prev_high - prev_low) * 1.1 / 2
    r3 = prev_close + camarilla_range
    s3 = prev_close - camarilla_range
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike (24-period on 12h ≈ 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike and price above daily EMA34
            if (close[i] > r3_12h[i] and 
                vol_spike[i] and
                close[i] > ema_34_12h[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Break below S3 with volume spike and price below daily EMA34
            elif (close[i] < s3_12h[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_12h[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below daily EMA34 (trend change)
            if close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above daily EMA34 (trend change)
            if close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals