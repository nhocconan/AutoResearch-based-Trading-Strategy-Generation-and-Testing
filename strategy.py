#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator strategy with 1w EMA50 trend filter and volume confirmation
# Long when Alligator Jaw < Teeth < Lips (bullish alignment) AND close > EMA50(1w) AND volume > 1.5x 20-period average
# Short when Alligator Jaw > Teeth > Lips (bearish alignment) AND close < EMA50(1w) AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks (jaws cross teeth or lips) OR EMA50(1w) trend flips
# Williams Alligator uses smoothed moving averages (SMA with specific periods) to identify trend
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume confirmation ensures breakout has institutional participation
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Discrete sizing (0.25) to limit fee drag

name = "1d_Williams_Alligator_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply smoothing (additional periods)
    jaw = smma(jaw_raw, 8) if len(jaw_raw) >= 8 else np.full_like(jaw_raw, np.nan, dtype=float)
    teeth = smma(teeth_raw, 5) if len(teeth_raw) >= 5 else np.full_like(teeth_raw, np.nan, dtype=float)
    lips = smma(lips_raw, 3) if len(lips_raw) >= 3 else np.full_like(lips_raw, np.nan, dtype=float)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish alignment (Jaw < Teeth < Lips) AND close > EMA50(1w) AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish alignment (Jaw > Teeth > Lips) AND close < EMA50(1w) AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (jaws cross teeth or lips) OR close < EMA50(1w) (trend flip)
            if (jaw_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= lips_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (jaws cross teeth or lips) OR close > EMA50(1w) (trend flip)
            if (jaw_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= lips_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals