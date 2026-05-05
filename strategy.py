#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks OR price crosses 1d EMA34
# Uses Williams Alligator for trend identification with smoothed SMAs (reduces whipsaw) and 1d EMA34 for higher timeframe trend confirmation
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing sustained trends in both bull and bear markets
# Timeframe: 12h (primary)
# Target symbols: BTC/ETH/SOL (avoid SOL-only bias)

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw (Blue line): 13-period SMMA smoothed by 8 periods
    # Teeth (Red line): 8-period SMMA smoothed by 5 periods  
    # Lips (Green line): 5-period SMMA smoothed by 3 periods
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(close_12h, 13)  # 13-period SMMA
    jaw = smma(jaw, 8)         # smoothed by 8 periods
    teeth = smma(close_12h, 8) # 8-period SMMA
    teeth = smma(teeth, 5)     # smoothed by 5 periods
    lips = smma(close_12h, 5)  # 5-period SMMA
    lips = smma(lips, 3)       # smoothed by 3 periods
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 12h (threshold: 1.5x for balanced filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish alignment (jaw < teeth < lips) AND price > EMA34 AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment (jaw > teeth > lips) AND price < EMA34 AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price < EMA34 (trend weakening)
            if not (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]) or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price > EMA34 (trend weakening)
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]) or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals