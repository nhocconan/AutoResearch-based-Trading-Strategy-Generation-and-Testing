#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1-week trend filter and volume confirmation.
# Long when: Jaw < Teeth < Lips (bullish alignment), price > Lips, weekly close > weekly EMA13, volume > 1.5x 20-day average
# Short when: Jaw > Teeth > Lips (bearish alignment), price < Lips, weekly close < weekly EMA13, volume > 1.5x 20-day average
# Exit when: alignment breaks or price crosses Jaw.
# Designed for ~10-20 trades/year per symbol. Works in both bull and bear markets by only taking trades when weekly trend aligns with daily Alligator.
name = "1d_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA13 for trend filter
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Williams Alligator on daily data
    # Jaw: 13-period SMMA smoothed 8 bars
    # Teeth: 8-period SMMA smoothed 5 bars
    # Lips: 5-period SMMA smoothed 3 bars
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA for different periods
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing shifts
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Alligator lines (already daily, no HTF alignment needed)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Volume average (20-day) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_close = close_1w[i // 7] if i >= 7 else np.nan  # Simplified weekly close approximation
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Check Alligator alignment
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: bullish alignment, price above Lips, weekly trend up, volume confirmation
            if (bullish_alignment and price > lips_aligned[i] and 
                weekly_close > ema13_1w_aligned[i] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below Lips, weekly trend down, volume confirmation
            elif (bearish_alignment and price < lips_aligned[i] and 
                  weekly_close < ema13_1w_aligned[i] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: alignment breaks or price crosses Jaw
            if not bullish_alignment or price < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks or price crosses Jaw
            if not bearish_alignment or price > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals