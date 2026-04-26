#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakouts with weekly EMA50 trend filter and volume spike. Targets 15-25 trades/year by requiring confluence of weekly trend alignment, volume confirmation, and price breaking key Camarilla levels (R3/S3). Uses discrete position sizing (0.30) to minimize fee churn while maintaining sufficient trade frequency. Weekly trend filter prevents counter-trend entries in choppy markets, and volume spike ensures institutional participation. Works in both bull and bear via weekly trend filter that adapts to long-term market direction.
Primary timeframe: 1d, HTF: 1w for trend filter.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Camarilla levels (R3, S3)
    # Camarilla: based on previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].values  # same as close_1w
    
    # Previous week's values (shifted by 1)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    # First week: use same values (will be filtered by min_periods later)
    high_1w_prev[0] = high_1w[0]
    low_1w_prev[0] = low_1w[0]
    close_1w_prev[0] = close_1w[0]
    
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    camarilla_range = high_1w_prev - low_1w_prev
    r3 = close_1w_prev + camarilla_range * 1.1 / 4
    s3 = close_1w_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to daily (no extra delay needed as they're based on completed weekly candles)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume spike: volume > 2.5x 50-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_50 = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.5 * vol_median_50)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for weekly EMA, 50 for volume median
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_median_50[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike and weekly uptrend (close > weekly EMA50)
            long_entry = (close_val > r3_aligned[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price breaks below S3 with volume spike and weekly downtrend (close < weekly EMA50)
            short_entry = (close_val < s3_aligned[i]) and vol_spike and (close_val < ema_50_val)
            
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
            # Long - exit on weekly trend reversal or price re-enters Camarilla (below S3)
            if close_val < ema_50_val or close_val < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on weekly trend reversal or price re-enters Camarilla (above R3)
            if close_val > ema_50_val or close_val > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0