#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R3/S3) from 1d for breakout entries, filtered by 1w EMA50 trend and volume spike (>2.0x average). In uptrend (price > EMA50), long on break above R3; in downtrend (price < EMA50), short on break below S3. Volume confirmation avoids false breakouts. Uses 1d timeframe with tight entries to avoid fee drag: target 7-25 trades/year. Camarilla levels provide strong support/resistance; EMA50 filters higher-timeframe trend; volume confirmation ensures conviction. Works in bull markets via trend-following breaks and in bear markets via shorting breakdowns.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # Warmup: need 1w EMA50 (50), 1d Camarilla (1), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_1w_val = ema_50_1w_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
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
            # Exit long: price returns to EMA50 or breaks below S3 (stoploss)
            exit_condition = (close_val < ema_1w_val) or (low_val < s3_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to EMA50 or breaks above R3 (stoploss)
            exit_condition = (close_val > ema_1w_val) or (high_val > r3_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0