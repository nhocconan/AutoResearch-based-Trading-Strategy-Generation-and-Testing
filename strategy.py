#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 1h timeframe with 4h trend alignment and volume confirmation capture high-probability intraday moves. 
4h EMA34 provides trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend). Volume > 1.5x 20-period average confirms momentum. 
Discrete sizing (0.20) limits fee drag. Target: 80-120 total trades over 4 years (20-30/year) for 1h timeframe.
Session filter (08-20 UTC) reduces noise during low-liquidity hours.
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
    
    # Get 4h data for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Camarilla levels (R3, S3) from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    range_4h = high_4h - low_4h
    camarilla_r3 = close_4h + 1.125 * range_4h
    camarilla_s3 = close_4h - 1.125 * range_4h
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (1h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm)
    
    # Session filter: 08-20 UTC (pre-compute hours from index)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20   # Position size: 20% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA34 (34), volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_4h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (4h)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
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
            # Exit conditions: stop at Camarilla S3 or reverse signal
            if close_val < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stop at Camarilla R3 or reverse signal
            if close_val > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0