#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
Hypothesis: Uses Camarilla pivot levels (R3, S3) on 1d timeframe for entry signals, filtered by 1d EMA34 trend and volume spikes. Designed for low trade frequency (~15-25 trades/year) on 12h timeframe to minimize fee drag. Works in both bull and bear markets by following the higher timeframe trend and entering on pullbacks to strong support/resistance levels.
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
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d candle
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3 = df_1d['close'] + range_val * 1.1 / 4
    s3 = df_1d['close'] - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (need previous day's levels)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3.values, additional_delay_bars=1)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3.values, additional_delay_bars=1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_level = r3_12h[i]
        s3_level = s3_12h[i]
        ema34 = ema34_12h[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3, above EMA34 trend, volume confirmation
            if price > r3_level and price > ema34 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below S3, below EMA34 trend, volume confirmation
            elif price < s3_level and price < ema34 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 (reversal signal)
            if price < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above R3 (reversal signal)
            if price > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0