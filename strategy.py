#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakout with daily trend filter and volume confirmation.
In daily uptrends (price > 1d EMA34), long on R3 breakout with volume spike.
In daily downtrends (price < 1d EMA34), short on S3 breakout with volume spike.
Uses Camarilla pivot levels from daily timeframe for precise support/resistance.
Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous daily bar
    # Using daily high, low, close from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from daily EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.8x average (strong confirmation)
        volume_confirm = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above R3 in uptrend regime with volume
            long_entry = (close[i] > r3_aligned[i]) and uptrend_regime and volume_confirm
            # Short: price breaks below S3 in downtrend regime with volume
            short_entry = (close[i] < s3_aligned[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or trend changes to downtrend
            if (close[i] < pivot[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above pivot or trend changes to uptrend
            if (close[i] > pivot[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals