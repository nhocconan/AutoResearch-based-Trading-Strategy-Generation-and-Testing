#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND close > 1d EMA50 AND volume > 2.0x 24-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND close < 1d EMA50 AND volume > 2.0x 24-bar avg
# Exit when Alligator alignment reverses (jaws-teeth-lips not in proper order)
# Uses discrete position sizing (0.25) to manage drawdown. Target: 12-30 trades/year on 6h.
# Williams Alligator identifies trending vs ranging markets via smoothed medians (Jaws=13, Teeth=8, Lips=5).
# In 2022 bear market, Alligator often remains bear-aligned, allowing shorts during rallies.
# Volume spike requirement filters for institutional participation, reducing whipsaws.
# 6h timeframe balances responsiveness with low trade frequency to overcome fee drag.

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 and Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50 and Alligator
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 1d:
    # Jaws: Smoothed Median (13, 8) -> 13-period SMMA of median, shifted 8 bars
    # Teeth: Smoothed Median (8, 5) -> 8-period SMMA of median, shifted 5 bars
    # Lips: Smoothed Median (5, 3) -> 5-period SMMA of median, shifted 3 bars
    # Where SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing
    median_1d = (high + low) / 2  # Using high/low from 1d data
    
    # Calculate SMMA using EMA as approximation (common practice)
    smma_13 = pd.Series(median_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    smma_8 = pd.Series(median_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    smma_5 = pd.Series(median_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply shifts (Alligator specific)
    jaws_1d = np.roll(smma_13, 8)  # Shifted 8 bars forward
    teeth_1d = np.roll(smma_8, 5)  # Shifted 5 bars forward
    lips_1d = np.roll(smma_5, 3)   # Shifted 3 bars forward
    
    # Handle NaN from roll
    jaws_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    jaws_1d_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: >2.0x 24-bar average volume (strict filter for low trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24, 13)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaws_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        jaws = jaws_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = jaws < teeth < lips  # Jaws lowest, then teeth, then lips highest
        bearish_alignment = jaws > teeth > lips  # Jaws highest, then teeth, then lips lowest
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish alignment AND price > lips AND close > 1d EMA50 AND volume confirmation
            if bullish_alignment and curr_close > lips and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < lips AND close < 1d EMA50 AND volume confirmation
            elif bearish_alignment and curr_close < lips and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when bullish alignment breaks
            if not (jaws < teeth < lips):  # Alignment broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when bearish alignment breaks
            if not (jaws > teeth > lips):  # Alignment broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals