#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_Volume_v1
Hypothesis: Camarilla pivot breakouts with 1d trend filter and volume spike capture strong trending moves while minimizing whipsaws. Designed for low trade frequency (~30-50/year) to reduce fee drag and improve robustness across bull and bear markets.
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
    
    # Calculate Camarilla levels from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: R3, R4, S3, S4
    range_prev = prev_high - prev_low
    camarilla_r4 = prev_close + range_prev * 1.1 / 2
    camarilla_r3 = prev_close + range_prev * 1.1 / 4
    camarilla_s3 = prev_close - range_prev * 1.1 / 4
    camarilla_s4 = prev_close - range_prev * 1.1 / 2
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Camarilla (1), EMA34 (34), volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        r3 = camarilla_r3[i]
        r4 = camarilla_r4[i]
        s3 = camarilla_s3[i]
        s4 = camarilla_s4[i]
        
        if position == 0:
            # Determine trend: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
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
            if close_val < r3:  # Re-enter Camarilla range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters R3-S3 range or trend reversal
            if close_val > s3:  # Re-enter Camarilla range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0