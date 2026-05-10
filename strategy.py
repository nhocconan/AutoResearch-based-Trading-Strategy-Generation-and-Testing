#!/usr/bin/env python3
# 1H_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
# Hypothesis: Uses 1h timeframe with 4h timeframe for trend confirmation and 1h for entry timing.
# Enters long when price breaks above 4h R3 in uptrend (4h close > 4h EMA50) with volume > 2x 20-period average.
# Enters short when price breaks below 4h S3 in downtrend (4h close < 4h EMA50) with volume confirmation.
# Exits when price returns to opposite level (S3 for long, R3 for short) or 4h trend reverses.
# Uses 4h EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 15-37 trades per year on 1h timeframe with position size 0.20 to minimize fee drag.
# Added session filter (08-20 UTC) to reduce noise trades.

name = "1H_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla pivots and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r3_level = prev_close + 1.1 * pivot_range
    s3_level = prev_close - 1.1 * pivot_range
    
    # Align pivot levels to 1h timeframe (available after 4h bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_level)
    
    # Volume filter: volume > 2x 20-period average on 1h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            if bars_since_entry >= 2 and in_session:
                # Long entry: price breaks above R3 in uptrend with volume spike
                if (close[i] > r3_aligned[i] and 
                    price_above_ema and 
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.20
                    position = 1
                    bars_since_entry = 0
                # Short entry: price breaks below S3 in downtrend with volume spike
                elif (close[i] < s3_aligned[i] and 
                      price_below_ema and 
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.20
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Long exit: price returns to S3 or trend reverses to downtrend
            if (close[i] < s3_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
                bars_since_entry += 1
        elif position == -1:
            # Short exit: price returns to R3 or trend reverses to uptrend
            if (close[i] > r3_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
                bars_since_entry += 1
    
    return signals