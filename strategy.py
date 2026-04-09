#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) + 1w EMA200 trend + volume confirmation
# Williams Alligator identifies trend absence/presence via smoothed medians (13,8,5 periods)
# 1w EMA200 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (1.5x 20-period avg) filters weak breakouts
# Works in bull/bear: EMA200 trend filter avoids ranging market failures
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25-0.30

name = "1d_1w_williams_alligator_ema200_volume_v1"
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
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend direction
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    median_price = (high + low) / 2
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # 13-period
    teeth = smma(median_price, 8)  # 8-period
    lips = smma(median_price, 5)   # 5-period
    
    # Shift as per Alligator specification: Jaw 8 bars, Teeth 5 bars, Lips 3 bars
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) >= 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) >= 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) >= 3:
        lips_shifted[3:] = lips[:-3]
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Alligator lines intertwine (no trend) OR price < 1w EMA200 (trend change)
            # Alligator sleeping: Jaw, Teeth, Lips are close together (intertwined)
            alligator_sleeping = (
                abs(jaw_shifted[i] - teeth_shifted[i]) < (close[i] * 0.001) and
                abs(teeth_shifted[i] - lips_shifted[i]) < (close[i] * 0.001) and
                abs(lips_shifted[i] - jaw_shifted[i]) < (close[i] * 0.001)
            )
            if alligator_sleeping or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines intertwine (no trend) OR price > 1w EMA200 (trend change)
            alligator_sleeping = (
                abs(jaw_shifted[i] - teeth_shifted[i]) < (close[i] * 0.001) and
                abs(teeth_shifted[i] - lips_shifted[i]) < (close[i] * 0.001) and
                abs(lips_shifted[i] - jaw_shifted[i]) < (close[i] * 0.001)
            )
            if alligator_sleeping or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Alligator alignment + 1w EMA200 trend filter
            if volume_confirmed:
                # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA200 (uptrend)
                if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                    close[i] > ema_200_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA200 (downtrend)
                elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                      close[i] < ema_200_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals