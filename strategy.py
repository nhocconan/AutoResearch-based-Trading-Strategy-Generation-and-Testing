#!/usr/bin/env python3
name = "12h_1d_Camarilla_R3S3_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Prices for Camarilla Calculation ===
    # Calculate daily high/low/close for Camarilla
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels from previous day
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align to 12h timeframe (values available at next 12h bar open)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Trend Filter: EMA34 ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Volume Spike Filter ===
    vol_avg_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume > (2.0 * vol_avg_12h)
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_12h[i]) or 
            np.isnan(camarilla_s3_12h[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above daily EMA34 + volume spike
            if (close[i] > camarilla_r3_12h[i] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 + below daily EMA34 + volume spike
            elif (close[i] < camarilla_s3_12h[i] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S3 (reversion) or below EMA34 (trend change)
            if close[i] < camarilla_s3_12h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R3 (reversion) or above EMA34 (trend change)
            if close[i] > camarilla_r3_12h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals