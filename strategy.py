#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3, S3) act as strong support/resistance derived from prior day's range.
Breaks above R3 or below S3 with volume spike indicate institutional participation. 
Trend filter using 1d EMA34 ensures trades align with higher timeframe direction: 
only long when price > 1d EMA34 (uptrend), short when price < 1d EMA34 (downtrend).
This avoids counter-trend whipsaws. 4h timeframe targets 20-50 trades/year (80-200 over 4 years).
Volume spike (>2x 20-bar average) confirms momentum validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    cam_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    cam_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align Camarilla levels to 4h timeframe (1-day delay for completed bar)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3.values)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND uptrend AND volume spike
            long_entry = (curr_high > cam_r3_aligned[i]) and uptrend and vol_spike
            # Short: price breaks below S3 AND downtrend AND volume spike
            short_entry = (curr_low < cam_s3_aligned[i]) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below S3 (failed breakout) OR loss of uptrend
            if (curr_low < cam_s3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above R3 (failed breakout) OR loss of downtrend
            if (curr_high > cam_r3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0