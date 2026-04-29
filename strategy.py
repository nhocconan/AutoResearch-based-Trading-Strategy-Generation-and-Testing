#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 Trend + Volume Spike
# Long when: Alligator Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when: Alligator Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when: Alligator lines crossover (Jaw crosses Teeth) OR price reverts to 1d EMA50
# Williams Alligator catches trends early with smoothed moving averages. 1d EMA50 filters counter-trend moves.
# Volume confirmation ensures breakout strength. Designed for 12h timeframe to avoid overtrading.
# Works in bull markets via trend following and in bear markets via short signals during downtrends.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h data (using median price)
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMA) - Williams Alligator uses SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator lines: Jaw (13-period SMMA, 8 bars offset), Teeth (8-period SMMA, 5 bars offset), Lips (5-period SMMA, 3 bars offset)
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for offset periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (already on 12h, no alignment needed)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator lines crossover (Jaw crosses below Teeth) OR price < EMA50
            if curr_jaw < curr_teeth or curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines crossover (Jaw crosses above Teeth) OR price > EMA50
            if curr_jaw > curr_teeth or curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish alignment: Jaw < Teeth < Lips
            bullish = curr_jaw < curr_teeth and curr_teeth < curr_lips
            # Bearish alignment: Jaw > Teeth > Lips
            bearish = curr_jaw > curr_teeth and curr_teeth > curr_lips
            
            # Long when bullish alignment AND price > 1d EMA50 AND volume confirmation
            if bullish and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1d EMA50 AND volume confirmation
            elif bearish and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals