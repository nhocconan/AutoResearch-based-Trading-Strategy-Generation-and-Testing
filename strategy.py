#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with weekly trend filter, volume confirmation, and ATR stop.
# Uses Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA).
# Long when Lips > Teeth > Jaw, price above Lips, weekly close > weekly open (bullish week), volume > 1.5x 20-day avg.
# Short when Lips < Teeth < Jaw, price below Lips, weekly close < weekly open (bearish week), volume > 1.5x 20-day avg.
# Exit when Alligator lines re-interlace or weekly trend reverses.
# Designed for 1d timeframe to capture multi-day trends with minimal whipsaw.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "1d_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    sma = np.mean(data[:period])
    smma_vals = np.full_like(data, np.nan, dtype=np.float64)
    smma_vals[period-1] = sma
    for i in range(period, len(data)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Williams Alligator components (5, 8, 13 periods)
    lips = smma(close, 5)      # SMMA(5)
    teeth = smma(close, 8)     # SMMA(8)
    jaw = smma(close, 13)      # SMMA(13)
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure all SMMA are ready (max period 13, need ~34 for stability)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment), price above Lips, bullish week, volume confirmation
            if (lips_val > teeth_val > jaw_val and 
                price > lips_val and 
                weekly_bull and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment), price below Lips, bearish week, volume confirmation
            elif (lips_val < teeth_val < jaw_val and 
                  price < lips_val and 
                  weekly_bear and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines re-interlace OR weekly trend turns bearish
            if not (lips_val > teeth_val > jaw_val) or weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines re-interlace OR weekly trend turns bullish
            if not (lips_val < teeth_val < jaw_val) or weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals