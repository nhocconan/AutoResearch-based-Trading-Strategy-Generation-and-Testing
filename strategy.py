#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
- Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Long when price breaks above R3 with volume spike and 1d uptrend
- Short when price breaks below S3 with volume spike and 1d downtrend
- Uses discrete position sizing (0.25) to minimize fee churn
- Designed to work in both bull (breakouts with trend) and bear (mean reversion at extremes)
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid NaN on first bar
    prev_close = np.where(np.isnan(prev_close), close_1d, prev_close)
    prev_high = np.where(np.isnan(prev_high), close_1d, prev_high)
    prev_low = np.where(np.isnan(prev_low), close_1d, prev_low)
    
    camarilla_range = (prev_high - prev_low) * 1.1 / 4.0
    r3 = prev_close + camarilla_range
    s3 = prev_close - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 1 for Camarilla, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND 1d uptrend
            if price_above_r3 and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND 1d downtrend
            elif price_below_s3 and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 OR 1d trend turns down
            if price_below_s3 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 OR 1d trend turns up
            if price_above_r3 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0