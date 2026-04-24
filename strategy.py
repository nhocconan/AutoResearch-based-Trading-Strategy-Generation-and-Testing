#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla Pivot Breakout with 1w EMA50 Trend Filter and Volume Spike.
- Long when price breaks above Camarilla R3 AND 1w EMA50 is rising AND volume > 2.0 * 20-period average
- Short when price breaks below Camarilla S3 AND 1w EMA50 is falling AND volume > 2.0 * 20-period average
- Exit when price returns to Camarilla pivot (PP) or volume drops below average
- Uses 1d primary with 1w HTF for trend filter to capture multi-week moves while avoiding counter-trend trades
- Camarilla levels provide precise intraday/resistance levels; EMA50 filter ensures direction alignment with weekly trend
- Volume spike confirms institutional participation and reduces false breakouts
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year)
"""

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
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # EMA50 slope: rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_1w_aligned, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w_aligned, dtype=bool)
    ema_50_rising[1:] = ema_50_1w_aligned[1:] > ema_50_1w_aligned[:-1]
    ema_50_falling[1:] = ema_50_1w_aligned[1:] < ema_50_1w_aligned[:-1]
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1  # Need volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND EMA50 rising AND volume spike
            if close[i] > camarilla_r3[i] and ema_50_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND EMA50 falling AND volume spike
            elif close[i] < camarilla_s3[i] and ema_50_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot OR volume drops below average
            if close[i] <= pivot[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot OR volume drops below average
            if close[i] >= pivot[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0