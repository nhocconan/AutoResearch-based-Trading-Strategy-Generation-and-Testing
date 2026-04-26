#!/usr/bin/env python3
"""
12h_WeeklyCamarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Weekly Camarilla R3/S3 breakouts on 12h with 1w EMA34 trend filter and volume confirmation.
Targets 12-37 trades/year by requiring confluence of weekly trend, volume spike, and Camarilla breakout.
Works in bull/bear markets via 1w trend filter (EMA34). Uses discrete position sizing (0.25) to minimize fee drag.
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
    open_time = prices['open_time'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels (R3, S3) from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align weekly Camarilla levels to 12h timeframe (no extra delay needed for pivot levels)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Volume spike: volume > 2.0x 24-period median volume (24x12h = 12d lookback)
    volume_series = pd.Series(volume)
    vol_median_24 = volume_series.rolling(window=24, min_periods=24).median().values
    volume_spike = volume > (2.0 * vol_median_24)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1w EMA, 24 for volume median
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_r3_1w_aligned[i]) or
            np.isnan(camarilla_s3_1w_aligned[i]) or
            np.isnan(vol_median_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        camarilla_r3 = camarilla_r3_1w_aligned[i]
        camarilla_s3 = camarilla_s3_1w_aligned[i]
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike and weekly uptrend (close > EMA34_1w)
            long_entry = (close_val > camarilla_r3) and vol_spike and (close_val > ema_34_val)
            # Short: price breaks below S3 with volume spike and weekly downtrend (close < EMA34_1w)
            short_entry = (close_val < camarilla_s3) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Camarilla range
            if close_val < ema_34_val or (close_val < camarilla_r3 and close_val > camarilla_s3):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Camarilla range
            if close_val > ema_34_val or (close_val < camarilla_r3 and close_val > camarilla_s3):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WeeklyCamarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0