#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v5
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1, close > 1d EMA34, and volume > 2.0x 20-period MA.
Enters short when price breaks below S1, close < 1d EMA34, and volume > 2.0x 20-period MA.
Exits when price reverts to Camarilla PP (pivot point) or opposite breakout occurs.
Uses 4h primary timeframe targeting 20-50 trades/year (80-200 total over 4 years).
Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters trend direction.
Volume spike confirms institutional participation. Works in bull/bear markets by aligning with
1d trend to avoid counter-trend trades and reduce whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's high, low, close
        if i < 1:
            # Not enough data for previous day
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Find previous day's bar index in 1d data
        current_time = prices.iloc[i]['open_time']
        prev_day = current_time - pd.Timedelta(days=1)
        
        # Get previous day's OHLC from 1d data
        mask = df_1d['open_time'] <= prev_day
        if not mask.any():
            # No previous day data yet
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        prev_day_idx = mask.sum() - 1
        if prev_day_idx < 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        prev_high = df_1d.iloc[prev_day_idx]['high']
        prev_low = df_1d.iloc[prev_day_idx]['low']
        prev_close = df_1d.iloc[prev_day_idx]['close']
        
        # Calculate Camarilla levels
        range_ = prev_high - prev_low
        if range_ <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla R1, S1, PP (Pivot Point)
        R1 = prev_close + (range_ * 1.1 / 12)
        S1 = prev_close - (range_ * 1.1 / 12)
        PP = (prev_high + prev_low + prev_close) / 3
        
        if position == 0:
            # Long: price breaks above R1, close > 1d EMA34, volume spike
            if (close[i] > R1 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, close < 1d EMA34, volume spike
            elif (close[i] < S1 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reverts to PP or breaks below S1 (failed breakout)
            if (close[i] <= PP or close[i] < S1):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reverts to PP or breaks above R1 (failed breakout)
            if (close[i] >= PP or close[i] > R1):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v5"
timeframe = "4h"
leverage = 1.0