#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) with 6h volume confirmation.
# Long when price breaks above R4 with volume spike (>1.8x median) - continuation breakout.
# Short when price breaks below S4 with volume spike - continuation breakdown.
# Fade trades: Long when price touches S3 with bullish 6h candle close, Short when price touches R3 with bearish 6h candle close.
# Uses discrete position size 0.25. Camarilla levels from 1d provide institutional support/resistance.
# Volume confirmation reduces false breakouts. Works in both bull/bear by trading with institutional levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels ===
    # Camarilla formula: based on previous day's range
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    camarilla_pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        high_y = high_1d[i-1]
        low_y = low_1d[i-1]
        close_y = close_1d[i-1]
        
        # Pivot point
        camarilla_pivot[i] = (high_y + low_y + close_y) / 3
        
        # Range
        range_y = high_y - low_y
        
        # Camarilla levels
        camarilla_r4[i] = camarilla_pivot[i] + (range_y * 1.1 / 2)
        camarilla_r3[i] = camarilla_pivot[i] + (range_y * 1.1 / 4)
        camarilla_s3[i] = camarilla_pivot[i] - (range_y * 1.1 / 4)
        camarilla_s4[i] = camarilla_pivot[i] - (range_y * 1.1 / 2)
    
    # Get 6h data for volume and price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 6h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(2, 20)  # Camarilla needs 1 day history, volume median 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        vol_median = vol_median_aligned[i]
        price = close[i]
        
        # Get current 6h volume for volume spike filter
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        current_vol_6h = vol_6h_aligned[i]
        
        # Volume spike filter: current 6h volume > 1.8x median volume
        volume_spike = current_vol_6h > (vol_median * 1.8)
        
        # 6h candle close direction for fade logic
        close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
        open_6h_aligned = align_htf_to_ltf(prices, df_6h, df_6h['open'].values)
        close_6h_price = close_6h_aligned[i]
        open_6h_price = open_6h_aligned[i]
        is_bullish_6h = close_6h_price > open_6h_price
        is_bearish_6h = close_6h_price < open_6h_price
        
        # === EXIT LOGIC: Flip position on opposite signal ===
        exit_long = False
        exit_short = False
        
        # Exit long if short signal triggers
        if (price < s3 and volume_spike and is_bearish_6h) or (price < s4 and volume_spike):
            exit_long = True
        
        # Exit short if long signal triggers
        if (price > r3 and volume_spike and is_bullish_6h) or (price > r4 and volume_spike):
            exit_short = True
        
        if exit_long and position == 1:
            signals[i] = 0.0
            position = 0
            continue
            
        if exit_short and position == -1:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # BREAKOUT ENTRIES (continuation)
            # LONG: price breaks above R4 with volume spike
            if price > r4 and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below S4 with volume spike
            elif price < s4 and volume_spike:
                signals[i] = -0.25
                position = -1
            
            # FADE ENTRIES (mean reversion at S3/R3)
            # LONG: price touches S3 with bullish 6h candle
            elif abs(price - s3) < (r4 - s4) * 0.02 and volume_spike and is_bullish_6h:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price touches R3 with bearish 6h candle
            elif abs(price - r3) < (r4 - s4) * 0.02 and volume_spike and is_bearish_6h:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dCamarilla_R3S3_R4S4_VolumeSpike1.8x_FadeBreakout_v1"
timeframe = "6h"
leverage = 1.0