#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R3/S3) from 1d for breakout entries on 12h timeframe, filtered by 1d EMA34 trend and volume spike (>2.0x average). In uptrend (price > EMA34), long on break above R3; in downtrend (price < EMA34), short on break below S3. Volume confirmation avoids false breakouts. Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag. Works in bull markets via trend-following breaks and in bear markets via shorting breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
    #            S3 = close - ((high-low) * 1.1/4), S4 = close - ((high-low) * 1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_factor = (high_1d - low_1d) * 1.1
    r3_1d = close_1d + (camarilla_factor / 4)
    s3_1d = close_1d - (camarilla_factor / 4)
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long on break above R3 with volume confirmation
                if (high_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short on break below S3 with volume confirmation
                if (low_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price returns to EMA34 or breaks below S3 (stoploss)
            exit_condition = (close_val < ema_1d_val) or (low_val < s3_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to EMA34 or breaks above R3 (stoploss)
            exit_condition = (close_val > ema_1d_val) or (high_val > r3_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0