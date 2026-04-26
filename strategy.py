#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h with 1w EMA50 trend filter and volume spike (>2.5x average volume).
R3/S3 are stronger reversal/continuation levels than R1/S1, leading to fewer but higher-quality trades.
In bull markets: price breaks above R3 with 1w uptrend and high volume → long.
In bear markets: price breaks below S3 with 1w downtrend and high volume → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
Requires BTC/ETH edge via 1w trend and volume filters; avoids SOL-only bias by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for EMA and Camarilla
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Camarilla calculation, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Need previous period's OHLC for Camarilla levels
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels (stronger levels)
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.5x average volume (stricter for fewer trades)
        volume_confirmed = vol > 2.5 * avg_vol
        
        # Long logic: price breaks above R3 with 1w uptrend and volume confirmation
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S3 with 1w downtrend and volume confirmation
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite Camarilla level break
        exit_long = (close_val < ema_val) or (close_val < s3)
        exit_short = (close_val > ema_val) or (close_val > r3)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0