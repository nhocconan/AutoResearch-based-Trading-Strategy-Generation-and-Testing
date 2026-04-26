#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA34 trend filter and volume spike confirmation.
Uses higher timeframe (1d) for trend alignment to reduce noise and false breakouts.
Targets 50-150 total trades over 4 years by requiring confluence of 1d trend, volume spike, and price breaking Camarilla levels.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
Primary timeframe: 12h, HTF: 1d for trend and Camarilla calculation.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla levels (R3, S3, R4, S4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous 1d candle's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as entry levels, R4/S4 as stop/profit targets (but we'll use trend for exit)
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.125 * range_1d
    s3 = close_1d - 1.125 * range_1d
    r4 = close_1d + 1.5 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h (no extra delay needed as they're based on completed 1d candles)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume median
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike and uptrend (close > EMA34_1d)
            long_entry = (close_val > r3_aligned[i]) and vol_spike and (close_val > ema_34_val)
            # Short: price breaks below S3 with volume spike and downtrend (close < EMA34_1d)
            short_entry = (close_val < s3_aligned[i]) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price reaches R4 (profit target) or re-enters S3
            if close_val < ema_34_val or close_val >= r4_aligned[i] or close_val < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price reaches S4 (profit target) or re-enters R3
            if close_val > ema_34_val or close_val <= s4_aligned[i] or close_val > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm"
timeframe = "12h"
leverage = 1.0