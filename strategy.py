#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakouts on 1h aligned with 4h EMA50 trend and volume spikes capture high-probability moves in both bull and bear markets. 
4h trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws. Volume confirmation adds conviction. 
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla levels (from prior 1h bar)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Calculate 1h Camarilla levels (R3, S3) from prior 1h bar
    # Shift by 1 to use prior bar's high/low/close
    high_1h_prev = np.roll(high_1h, 1)
    low_1h_prev = np.roll(low_1h, 1)
    close_1h_prev = np.roll(close_1h, 1)
    high_1h_prev[0] = high_1h[0]  # first bar: use current
    low_1h_prev[0] = low_1h[0]
    close_1h_prev[0] = close_1h[0]
    range_1h = high_1h_prev - low_1h_prev
    camarilla_r3 = close_1h_prev + 1.125 * range_1h
    camarilla_s3 = close_1h_prev - 1.125 * range_1h
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (1h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20   # Position size: 20% of capital (discrete level to reduce fee drag)
    
    # Warmup: need Camarilla (1), EMA50 (50), volume avg (20)
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        ema50 = ema50_4h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend alignment: price vs 4h EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above R3 with volume
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below S3 with volume
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla S3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla R3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0