#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, trade long when price breaks above Camarilla R3 level with volume spike and above 12h EMA50 trend, short when breaks below S3 with volume spike and below 12h EMA50. Uses discrete sizing (0.25) to limit fee drag. Camarilla R3/S3 are stronger breakout levels than R1/S1, reducing false signals. Volume spike filter confirms momentum. 12h EMA50 trend filter ensures alignment with medium-term trend. ATR-based stoploss (2*ATR) manages risk. Designed to work in both bull and bear markets by aligning with 12h trend and using volatility-based stops.
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
    
    # Get 12h data for EMA50 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range for prior 12h bar
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = close_12h[0]  # first bar
    tr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - prev_close_12h), np.abs(low_12h - prev_close_12h)))
    atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior bar's range (R3/S3 are stronger breakout levels)
    hl_range_12h = high_12h - low_12h
    r3_12h = close_12h + 1.1666 * hl_range_12h  # R3 level
    s3_12h = close_12h - 1.1666 * hl_range_12h  # S3 level
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # ATR for stoploss calculation (4h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_12h_aligned[i]
        r3_val = r3_12h_aligned[i]
        s3_val = s3_12h_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R3, above 12h EMA50, with volume spike
            long_signal = (close_val > r3_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below S3, below 12h EMA50, with volume spike
            short_signal = (close_val < s3_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR ATR stoploss (2*ATR below entry)
            if (close_val < s3_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR ATR stoploss (2*ATR above entry)
            if (close_val > r3_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0