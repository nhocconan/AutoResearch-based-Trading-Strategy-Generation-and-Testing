#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and 1d volume confirmation.
- Uses 1h Camarilla pivot levels (R3/S3) for precise breakout entries
- 4h EMA50 defines intermediate trend: only trade breakouts in 4h trend direction
- 1d volume spike (>2.0x 20-period average) confirms institutional participation
- Session filter (08-20 UTC) reduces noise trades outside active hours
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by trading with the 4h trend
- Discrete position sizing (0.20) minimizes fee churn
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
    open_time = prices['open_time'].values
    
    # Calculate 1h Camarilla pivot levels (R3, S3)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_1h = close_1h + (high_1h - low_1h) * 1.1 / 4
    camarilla_s3_1h = close_1h - (high_1h - low_1h) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3_1h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3_1h)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume confirmation (>2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, vol_ma_1d * 2.0)  # threshold = 2.0x MA
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_spike_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 with 4h uptrend and 1d volume spike
            long_breakout = (close[i] > camarilla_r3_aligned[i] and 
                           close[i] > ema_50_4h_aligned[i] and
                           volume[i] > vol_spike_1d[i])
            
            # Short conditions: price breaks below Camarilla S3 with 4h downtrend and 1d volume spike
            short_breakout = (close[i] < camarilla_s3_aligned[i] and 
                            close[i] < ema_50_4h_aligned[i] and
                            volume[i] > vol_spike_1d[i])
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or 4h trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 or 4h trend turns bearish
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R3 or 4h trend turns bullish
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_1dVolumeSpike_Session"
timeframe = "1h"
leverage = 1.0