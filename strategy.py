#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 12h timeframe filtered by 1d EMA34 trend and volume spikes. Designed for low-frequency, high-conviction trades (12-37/year) with discrete sizing to minimize fee drag. Works in bull markets via breakouts above R3 in uptrend, and in bear markets via breakdowns below S3 in downtrend.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Camarilla levels from previous completed 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (strict threshold to avoid overtrading)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need 1d EMA34 (34), 1d shift(1), vol avg (24)
    start_idx = max(34 + 1*12, 24)  # ~34*12 + 24 = 432 bars for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1d EMA34 alignment and volume confirmation
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0