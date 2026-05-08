#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend Filter and Volume Spike
# - Williams Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset)
# - Long when Lips > Teeth > Jaw (bullish alignment) + price above Lips + volume spike
# - Short when Lips < Teeth < Jaw (bearish alignment) + price below Lips + volume spike
# - Uses 1d EMA50 trend filter to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 12h timeframe

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=np.float64)
    result = np.full_like(data, np.nan, dtype=np.float64)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price_12h = (high_12h := (df_12h['high'].values + df_12h['low'].values) / 2.0)
    
    # Jaw: 13-period SMMA of median price, 8 bars offset
    jaw_raw = smma(median_price_12h, 13)
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid due to offset
    
    # Teeth: 8-period SMMA of median price, 5 bars offset
    teeth_raw = smma(median_price_12h, 8)
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid due to offset
    
    # Lips: 5-period SMMA of median price, 3 bars offset
    lips_raw = smma(median_price_12h, 5)
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid due to offset
    
    # Align Alligator components to 12h timeframe (already on 12h, but ensure proper alignment)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume spike: current volume > 1.5x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price above Lips + 1d uptrend + volume spike
            long_cond = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                        close[i] > lips_aligned[i] and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Lips < Teeth < Jaw (bearish alignment) + price below Lips + 1d downtrend + volume spike
            short_cond = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                         close[i] < lips_aligned[i] and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips < Teeth (bullish alignment broken) or price below Jaw
            if lips_aligned[i] < teeth_aligned[i] or close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips > Teeth (bearish alignment broken) or price above Jaw
            if lips_aligned[i] > teeth_aligned[i] or close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals