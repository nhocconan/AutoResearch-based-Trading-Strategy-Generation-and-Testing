#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: 1-hour Camarilla R1/S1 breakout with 4-hour EMA50 trend filter and volume spike confirmation.
Uses 4h for signal direction (trend + Camarilla levels) and 1h only for entry timing precision.
Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Volume spike (>2x average) confirms breakout strength. Works in bull/bear via 4h EMA50 trend filter.
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
    
    # Get 4h data for HTF filters (trend and Camarilla levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(50), ATR(14), volume MA(20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        vol_spike = vol_ratio[i] > 2.0  # volume at least 2x average
        trend_4h_up = close_val > ema_50_4h_aligned[i]
        trend_4h_down = close_val < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 4h trend up AND volume spike
            long_signal = (close_val > camarilla_r1_aligned[i]) and trend_4h_up and vol_spike
            
            # Short: price breaks below Camarilla S1 AND 4h trend down AND volume spike
            short_signal = (close_val < camarilla_s1_aligned[i]) and trend_4h_down and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_4h_up) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_4h_down) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0