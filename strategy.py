#!/usr/bin/env python3
"""
12h_1D_Camarilla_R3_S3_Breakout_1D_Trend_Volume_Spike
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout detection,
combined with 1-day trend (close > EMA34) and volume spikes for confirmation.
Designed for low-frequency, high-conviction trades on 12H timeframe to minimize fee drag.
Targets 12-37 trades per year (~50-150 over 4 years) for robustness in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels from previous day
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    # where C, H, L are from previous completed day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 12H timeframe (waits for daily close)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12H INDICATORS ===
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA34 and volume average
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_12h[i]
        s3_val = s3_12h[i]
        ema34_val = ema34_12h[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long conditions: break above R3, above EMA34 trend, volume confirmation
            if close_val > r3_val and close_val > ema34_val and vol_conf:
                signals[i] = size
                position = 1
            # Short conditions: break below S3, below EMA34 trend, volume confirmation
            elif close_val < s3_val and close_val < ema34_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (reversal signal) or trend fails
            if close_val < s3_val or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R3 (reversal signal) or trend fails
            if close_val > r3_val or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_1D_Camarilla_R3_S3_Breakout_1D_Trend_Volume_Spike"
timeframe = "12h"
leverage = 1.0