#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) for structure, 4h EMA50 for trend alignment to avoid counter-trend entries
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.20) minimizes fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) for 1h timeframe
# Session filter (08-20 UTC) reduces noise trades
# Works in bull markets via upper Camarilla breakout continuation and in bear markets via lower Camarilla breakdown continuation

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Camarilla levels (R3, S3) on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (no extra delay needed as they're based on completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike (2.0x 20-period average) on 1h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and Camarilla calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 4h close > EMA50 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + 4h close < EMA50 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 or 4h trend breaks
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 or 4h trend breaks
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals