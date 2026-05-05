#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation (2.0x)
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks OR 1d EMA50 filter reverses
# Williams Alligator identifies trend initiation and continuation with smoothed moving averages
# 1d EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Volume confirmation reduces false signals during low participation periods
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1d

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_2.0x"
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
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaws: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA values
    jaws_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply forward shifts (Alligator specific)
    jaws = np.roll(jaws_raw, 8)  # shifted 8 bars forward
    teeth = np.roll(teeth_raw, 5)  # shifted 5 bars forward
    lips = np.roll(lips_raw, 3)   # shifted 3 bars forward
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish alignment (jaws < teeth < lips) AND price > EMA50 AND volume spike
            if (jaws[i] < teeth[i] < lips[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment (jaws > teeth > lips) AND price < EMA50 AND volume spike
            elif (jaws[i] > teeth[i] > lips[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bullish alignment breaks OR price < EMA50 (trend weakening)
            if not (jaws[i] < teeth[i] < lips[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bearish alignment breaks OR price > EMA50 (trend weakening)
            if not (jaws[i] > teeth[i] > lips[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals