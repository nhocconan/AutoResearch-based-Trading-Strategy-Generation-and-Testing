#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakouts on 6h aligned with 12h EMA50 trend and volume spikes capture high-probability swing trades. 
The 12h timeframe filters noise and aligns with institutional swing trading cycles. Volume confirmation ensures breakout validity. 
Discrete sizing (0.25) controls risk and minimizes fee churn. Target: 75-200 total trades over 4 years.
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
    
    # Get 6h data for Camarilla levels
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    range_6h = high_6h - low_6h
    camarilla_r3 = close_6h + 1.125 * range_6h
    camarilla_s3 = close_6h - 1.125 * range_6h
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_6h, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA50 (50), volume avg (20)
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema50 = ema50_12h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA50 (12h)
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

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0