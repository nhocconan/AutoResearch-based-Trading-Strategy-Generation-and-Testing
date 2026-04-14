#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily volume confirmation and weekly trend filter
# Long when price > Alligator Jaw (teeth) with volume spike and weekly bullish trend
# Short when price < Alligator Jaw (teeth) with volume spike and weekly bearish trend
# Exit when price crosses Alligator Teeth
# Williams Alligator uses SMAs of 13, 8, 5 periods with future shifts (8,5,3) to avoid look-ahead
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams Alligator is trend-following but avoids whipsaw via smoothed SMAs with shifts
# Works in both bull (trend following) and bear (avoids counter-trend via weekly filter)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and weekly data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (Smoothed Moving Average)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for Alligator calculation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # Jaw (13-period SMA shifted 8)
    jaw_raw = pd.Series(typical_price_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth (8-period SMA shifted 5)
    teeth_raw = pd.Series(typical_price_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips (5-period SMA shifted 3)
    lips_raw = pd.Series(typical_price_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe (then to lower timeframe if needed)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 13,8,5,20,21 plus shifts)
    start = 50  # conservative start to avoid NaN issues
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current volume (using 12h equivalent)
        
        if position == 0:
            # Long setup: price > Teeth (Alligator teeth) with volume spike and weekly bullish trend
            # Alligator alignment: Lips > Teeth > Jaw = bullish
            if (price > teeth_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                vol_12h_current > 2.0 * vol_ma_12h_aligned[i] and  # Volume spike
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: price < Teeth (Alligator teeth) with volume spike and weekly bearish trend
            # Alligator alignment: Lips < Teeth < Jaw = bearish
            elif (price < teeth_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  vol_12h_current > 2.0 * vol_ma_12h_aligned[i] and  # Volume spike
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth or Alligator loses bullish alignment
            if (price < teeth_aligned[i] or 
                not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Teeth or Alligator loses bearish alignment
            if (price > teeth_aligned[i] or 
                not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0