#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R3S3_Breakout_1dTrend"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels: R3, S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot calculation: (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    r3_1d = high_1d + 2.0 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2.0 * (high_1d - pivot_1d)
    
    # Align Camarilla levels to 1h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1h volume filter: > 1.5x 20-period average
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma_1h
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if session_filter[i] and vol_filter[i]:
            if position == 0:
                # Long: Close > R3 with 4h uptrend
                if close[i] > r3_1d_aligned[i] and close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Close < S3 with 4h downtrend
                elif close[i] < s3_1d_aligned[i] and close[i] < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit: Close < EMA_4h (trend change) or close < S3 (mean reversion)
                if close[i] < ema_4h_aligned[i] or close[i] < s3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: Close > EMA_4h (trend change) or close > R3 (mean reversion)
                if close[i] > ema_4h_aligned[i] or close[i] > r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session or low volume: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(50) trend filter, volume confirmation, and session filter (08-20 UTC).
# Camarilla levels identify key support/resistance from daily price action.
# Breakout above R3 in uptrend (close > 4h EMA50) or below S3 in downtrend captures momentum.
# Volume filter ensures institutional participation. Session filter reduces noise.
# Target: 15-30 trades/year to minimize fee drag. Position size 0.20 limits risk.
# Works in both bull and bear markets by following 4h trend direction.