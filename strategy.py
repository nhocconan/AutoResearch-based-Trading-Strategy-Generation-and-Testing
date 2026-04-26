#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA200 trend filter and volume confirmation. In trending markets (price above/below 1w EMA200), buy breakouts above R1 or sell breakdowns below S1. Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (target: 30-100 trades over 4 years) to work in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for HTF trend
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    htf_trend = np.where(close > ema_200_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate volume spike filter (20-period volume SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_sma_20 * 2.0)  # Volume at least 2x average
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first values to avoid NaN
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + (camarilla_range * 1.1 / 12)
    s1 = prev_close - (camarilla_range * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 200 for EMA, 20 for volume SMA)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_20[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout logic with trend and volume confirmation
        if htf_trend[i] == 1 and volume_spike[i]:  # Uptrend + volume spike
            if close[i] > r1[i]:  # Break above R1
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < s1[i]:  # Break below S1 (exit long or go short only if counter-trend not allowed)
                # In uptrend, only exit longs, don't go short
                if position == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0  # Stay flat
        elif htf_trend[i] == -1 and volume_spike[i]:  # Downtrend + volume spike
            if close[i] < s1[i]:  # Break below S1
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif close[i] > r1[i]:  # Break above R1 (exit short or go long only if counter-trend not allowed)
                # In downtrend, only exit shorts, don't go long
                if position == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0  # Stay flat
        else:
            # No clear signal - hold current position or exit if price returns to Camarilla range
            if position == 1 and close[i] < r1[i] and close[i] > s1[i]:
                # Price back inside range - exit long
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > s1[i] and close[i] < r1[i]:
                # Price back inside range - exit short
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0