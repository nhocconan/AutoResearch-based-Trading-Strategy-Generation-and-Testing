#!/usr/bin/env python3
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
    
    # Get 4h data for trend and 1d data for pivots
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Daily EMA(34) for long-term trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Focus on R3/S3 for fading
    r3 = pivot + range_ * 1.25
    s3 = pivot - range_ * 1.25
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        ema_trend_4h = ema21_4h_aligned[i]
        ema_trend_1d = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Fade at R3/S3: price touches level and reverses
            # Long: touch S3, close above it, in uptrend (both 4h and 1d), volume spike
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and 
                close[i] > ema_trend_4h and close[i] > ema_trend_1d and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: touch R3, close below it, in downtrend (both 4h and 1d), volume spike
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and 
                  close[i] < ema_trend_4h and close[i] < ema_trend_1d and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S3 (mean reversion) or trend weakens
            if low[i] <= s3_aligned[i] or close[i] < ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R3 (mean reversion) or trend weakens
            if high[i] >= r3_aligned[i] or close[i] > ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3S3_Fade_4hEMA21_1dEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0