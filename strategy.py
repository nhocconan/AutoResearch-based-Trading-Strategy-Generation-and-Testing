#!/usr/bin/env python3
"""
12h Williams Alligator + 1d Trend + Volume Spike
Long when price above Alligator teeth (green line) with bullish 1d EMA and volume spike.
Short when price below Alligator teeth with bearish 1d EMA and volume spike.
Exit when price crosses back below/above teeth or trend weakens.
Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_d = pd.Series(df_daily['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        result = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                if not np.isnan(sma.iloc[i]):
                    result[i] = (result[i-1] * (period-1) + sma.iloc[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift jaws/teeth/lips as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after all lookbacks
        # Skip if data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above lips (green) with bullish 1d trend and volume spike
            if (close[i] > lips_shifted[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price below lips (green) with bearish 1d trend and volume spike
            elif (close[i] < lips_shifted[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below lips OR trend turns bearish
                if close[i] <= lips_shifted[i] or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above lips OR trend turns bullish
                if close[i] >= lips_shifted[i] or close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%