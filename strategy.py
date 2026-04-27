#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from daily data act as strong support/resistance.
Breaks of these levels with volume confirmation and 1-week trend filter capture
institutional interest-driven moves. Designed for low trade frequency (12-37/year)
to minimize fee drift in both bull and bear markets. Works in bull (breakouts continue)
and bear (false breaks reversed quickly by trend filter).
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
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from prior day
    # R3 = close + 1.1*(high-low)*1.1/2
    # S3 = close - 1.1*(high-low)*1.1/2
    # Using prior day's values to avoid look-ahead
    rang = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rang * 1.1 / 2
    camarilla_s3 = close_1d - 1.1 * rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (they update at 00:00 UTC daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1-week trend filter: EMA50 on weekly
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (2 days of 12h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d data (1), 1w EMA50 (50), volume avg (24)
    start_idx = max(1, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema50_1w = ema50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price vs weekly EMA50
            uptrend = close_val > ema50_1w
            downtrend = close_val < ema50_1w
            
            if uptrend and vol_conf:
                # Long: break above R3 with volume
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S3 with volume
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters R3-S3 range or trend reversal
            if close_val < r3:  # Re-enter below R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters R3-S3 range or trend reversal
            if close_val > s3:  # Re-enter above S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0