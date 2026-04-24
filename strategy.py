#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses 4h HTF for signal direction (Camarilla breakouts) and 1d EMA50 for trend filter to avoid whipsaws.
- 1h timeframe only for precise entry timing after HTF conditions are met.
- Volume spike filter (>2.0x median of last 20 bars) to avoid fakeouts.
- Session filter (08-20 UTC) to reduce noise during low-liquidity hours.
- Discrete position sizing (0.20) to limit fee churn and manage drawdown.
- Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits.
- Designed for BTC/ETH: Camarilla works in ranging markets, EMA50 filters trend, volume confirms momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour
    
    # Calculate Camarilla levels (based on previous bar's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Get 4h data ONCE before loop for Camarilla levels (signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    camarilla_r1_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 12
    camarilla_s1_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 12
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_4h_aligned[i]) or np.isnan(camarilla_s1_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R1, trend up (close > 1d EMA50), volume spike
            if close[i] > camarilla_r1_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S1, trend down (close < 1d EMA50), volume spike
            elif close[i] < camarilla_s1_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Camarilla S1 OR trend reversal (close < 1d EMA50)
            if close[i] < camarilla_s1_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Camarilla R1 OR trend reversal (close > 1d EMA50)
            if close[i] > camarilla_r1_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_1dEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0