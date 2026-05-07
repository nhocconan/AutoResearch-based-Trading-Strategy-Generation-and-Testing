#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: 6h Camarilla breakout at R3/S3 levels with 1d trend filter and volume confirmation.
# In bull markets (price > 1d EMA34), long on R3 breakout with volume.
# In bear markets (price < 1d EMA34), short on S3 breakout with volume.
# Uses Camarilla pivot levels from daily timeframe for structure, reducing false signals.
# Target: 50-150 total trades over 4 years (~12-37/year) with position size 0.25.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Calculate Camarilla levels
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    # S4 = close - (high - low) * 1.1
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 2
    s3 = prev_close - range_hl * 1.1 / 2
    r4 = prev_close + range_hl * 1.1
    s4 = prev_close - range_hl * 1.1
    
    # Align Camarilla levels to 6h timeframe (wait for previous day to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need 40 periods for sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above R3 in uptrend regime + volume
            long_entry = (close[i] > r3_aligned[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below S3 in downtrend regime + volume
            short_entry = (close[i] < s3_aligned[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below S3 or regime changes to downtrend
            if (close[i] < s3_aligned[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above R3 or regime changes to uptrend
            if (close[i] > r3_aligned[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals