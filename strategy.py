#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (SMA5,8,13) with daily trend filter and volume spike confirmation.
# Long when Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND daily trend bullish (price > weekly EMA50) AND volume > 1.5x daily average volume
# Short when Jaw > Teeth > Lips (bearish alignment) AND price < Jaw AND daily trend bearish (price < weekly EMA50) AND volume > 1.5x daily average volume
# Exit when Alligator lines cross (loss of alignment) or price crosses middle line (Teeth)
# Williams Alligator identifies trend phases; volume confirms strength; weekly EMA filters higher timeframe trend.
# Target: 12-30 trades/year per symbol.
name = "12h_Williams_Alligator_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator components (SMA5, SMA8, SMA13 on median price)
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw (13-period)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Teeth (8-period)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Lips (5-period)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get weekly data for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Get daily average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Ensure Alligator and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        weekly_ema = weekly_ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_alignment = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        # Price relative to Alligator lines
        price_above_lips = price > lips_val
        price_below_jaw = price < jaw_val
        
        # Daily trend filter from weekly EMA
        daily_bullish_trend = price > weekly_ema
        daily_bearish_trend = price < weekly_ema
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: bullish alignment + price above lips + bullish trend + volume confirmation
            if bullish_alignment and price_above_lips and daily_bullish_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + price below jaw + bearish trend + volume confirmation
            elif bearish_alignment and price_below_jaw and daily_bearish_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: loss of bullish alignment OR price crosses below teeth
            if not (jaw_val < teeth_val < lips_val) or price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: loss of bearish alignment OR price crosses above teeth
            if not (jaw_val > teeth_val > lips_val) or price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals