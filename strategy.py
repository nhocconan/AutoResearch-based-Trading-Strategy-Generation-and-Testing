#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA(50) trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) + volume spike + price > 1d EMA(50)
# Short when Alligator jaws > teeth > lips (bearish alignment) + volume spike + price < 1d EMA(50)
# Uses Williams Alligator (SMAs: jaws=13, teeth=8, lips=5) from previous 12h bar to avoid look-ahead
# 1d EMA(50) filter reduces whipsaw and captures medium-term trend
# Volume spike (1.8x 20-period average) confirms institutional participation
# Designed for low trade frequency (12-37/year on 12h) to minimize fee drag
# Works in both bull (trend continuation) and bear (trend reversal) markets

name = "12h_WilliamsAlligator_Volume_1dEMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    # Jaws: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (df_12h['high'] + df_12h['low']) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(median_price.values, 13)
    teeth = smma(median_price.values, 8)
    lips = smma(median_price.values, 5)
    
    # Align 12h Alligator lines to 12h timeframe (wait for completed 12h bar)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (1.8x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(13 for jaws, 20 for volume MA, 50 for 1d EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment (jaws < teeth < lips) + volume spike + price > 1d EMA(50)
            if (jaws_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and 
                volume_spike[i] and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment (jaws > teeth > lips) + volume spike + price < 1d EMA(50)
            elif (jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and 
                  volume_spike[i] and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment turns bearish OR price below 1d EMA(50)
            if (jaws_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > lips_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment turns bullish OR price above 1d EMA(50)
            if (jaws_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < lips_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals