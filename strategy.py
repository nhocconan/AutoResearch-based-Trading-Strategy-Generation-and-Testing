#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction via smoothed medians
# Breakout occurs when Lips cross above/below Teeth with price confirmation
# 1w EMA50 ensures alignment with long-term trend to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via upward Alligator alignment and bear markets via downward alignment.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator (requires high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three smoothed medians
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    # Teeth (red): 8-period SMMA, shifted 5 bars  
    # Lips (green): 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma((high_1d + low_1d) / 2, 13)  # Jaw uses median price
    teeth = smma((high_1d + low_1d) / 2, 8)  # Teeth uses median price
    lips = smma((high_1d + low_1d) / 2, 5)   # Lips uses median price
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 1d timeframe (no shift needed as already 1d)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Alligator alignment (Lips > Teeth > Jaw for bull, reverse for bear)
            if curr_volume_spike:
                # Bullish entry: Lips above Teeth above Jaw (bullish alignment) AND price > EMA50_1w
                if curr_lips > curr_teeth > curr_jaw and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips below Teeth below Jaw (bearish alignment) AND price < EMA50_1w
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator alignment breaks (Lips crosses below Teeth) OR price crosses below EMA50_1w
            if curr_lips < curr_teeth or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment breaks (Lips crosses above Teeth) OR price crosses above EMA50_1w
            if curr_lips > curr_teeth or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals