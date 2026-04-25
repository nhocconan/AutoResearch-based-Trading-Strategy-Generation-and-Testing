#!/usr/bin/env python3
"""
6h Camarilla R3/S3 Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. 
Breakouts beyond these levels with 1d EMA50 trend alignment and volume confirmation
capture continuation moves. Works in bull via breakout continuation and in bear 
via avoiding false breakouts. Discrete sizing (0.25) controls drawdown.
Target: 50-150 total trades over 4 years on 6h timeframe.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use the previous completed 1d bar to avoid look-ahead
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Get previous completed 1d bar indices
        # Since we're on 6h timeframe, we need to map to 1d bars
        # Use align_htf_to_ltf in reverse: get the 1d bar that completed before current 6h bar
        if i >= 4:  # 4*6h = 24h = 1d
            prev_1d_idx = (i // 4) - 1  # previous completed 1d bar
            if prev_1d_idx >= 0 and prev_1d_idx < len(df_1d):
                phigh = df_1d['high'].iloc[prev_1d_idx]
                plow = df_1d['low'].iloc[prev_1d_idx]
                pclose = df_1d['close'].iloc[prev_1d_idx]
                range_val = phigh - plow
                if range_val > 0:
                    camarilla_r3[i] = pclose + (range_val * 1.1 / 4)
                    camarilla_s3[i] = pclose - (range_val * 1.1 / 4)
                    camarilla_r4[i] = pclose + (range_val * 1.1 / 2)
                    camarilla_s4[i] = pclose - (range_val * 1.1 / 2)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_1d, volume MA, and Camarilla levels to propagate
    start_idx = max(50, 20, 4)  # 4 for Camarilla (need at least one prior 1d bar)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (close > 1d EMA50) AND volume spike
            long_condition = (curr_close > r3) and (curr_close > ema50_1d) and volume_spike
            # Short: price breaks below S3 AND downtrend (close < 1d EMA50) AND volume spike
            short_condition = (curr_close < s3) and (curr_close < ema50_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price breaks below S3 (reversal signal) or above R4 (take profit)
            if curr_close < s3 or curr_close > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (reversal signal) or below S4 (take profit)
            if curr_close > r3 or curr_close < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0