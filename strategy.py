#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w trend filter + volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# 1w EMA50 filter ensures we only trade in alignment with weekly trend
# Volume confirmation (1d volume > 1.5 * 20-period average) filters false breakouts
# Works in bull/bear: Alligator adapts to changing markets, weekly filter prevents counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_1w_williams_alligator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw: Blue line (13-period SMMA, shifted 8 bars)
    # Teeth: Red line (8-period SMMA, shifted 5 bars)
    # Lips: Green line (5-period SMMA, shifted 3 bars)
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Shift the lines (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    # Align 1w EMA50 to 1d timeframe (wait for 1w bar close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Alligator signals:
        # Lips above Teeth above Jaw = bullish alignment
        # Lips below Teeth below Jaw = bearish alignment
        # Intertwined lines = ranging/no trend
        bullish_alignment = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
        bearish_alignment = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
        
        # Weekly trend filter: only trade in direction of weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish alignment OR price crosses below Jaw
            if bearish_alignment or close[i] < jaw_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish alignment OR price crosses above Jaw
            if bullish_alignment or close[i] > jaw_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: trade in direction of weekly trend with Alligator alignment
            if weekly_uptrend and bullish_alignment and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif weekly_downtrend and bearish_alignment and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals