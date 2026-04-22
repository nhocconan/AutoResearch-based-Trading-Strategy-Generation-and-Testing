#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
Long when Alligator jaws (13-period smoothed median) is above teeth (8-period) and lips (5-period),
price above jaws, and volume spike with bullish 1d trend (price > 1d EMA34).
Short when jaws below teeth and lips, price below jaws, volume spike, and bearish 1d trend.
Exit when Alligator lines cross in opposite direction or trend weakens.
Designed for low trade frequency (20-40/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator calculation
    median_price = (high + low) / 2.0
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_d = pd.Series(df_daily['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate Williams Alligator components on median price
    # Lips: 5-period SMMA of median, shifted 3 bars forward
    lips = smma(median_price, 5)
    # Teeth: 8-period SMMA of median, shifted 5 bars forward
    teeth = smma(median_price, 8)
    # Jaw: 13-period SMMA of median, shifted 8 bars forward
    jaw = smma(median_price, 13)
    
    # Apply the forward shifts as per Alligator definition
    lips_shifted = np.roll(lips, 3)
    teeth_shifted = np.roll(teeth, 5)
    jaw_shifted = np.roll(jaw, 8)
    
    # Set NaN for shifted positions that would look ahead
    lips_shifted[:3] = np.nan
    teeth_shifted[:5] = np.nan
    jaw_shifted[:8] = np.nan
    
    # Align Alligator components to 4h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth_shifted)
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw_shifted)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Jaws > Teeth > Lips (bullish alignment), price above jaws, volume spike, bullish trend
            if (jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and
                close[i] > jaw_aligned[i] and
                volume[i] > 2.0 * vol_avg_20[i] and
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws < Teeth < Lips (bearish alignment), price below jaws, volume spike, bearish trend
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and
                  close[i] < jaw_aligned[i] and
                  volume[i] > 2.0 * vol_avg_20[i] and
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment breaks down (jaws < teeth) or price below jaws or trend turns bearish
                if (jaw_aligned[i] < teeth_aligned[i] or 
                    close[i] < jaw_aligned[i] or
                    close[i] < ema34_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment breaks down (jaws > teeth) or price above jaws or trend turns bullish
                if (jaw_aligned[i] > teeth_aligned[i] or 
                    close[i] > jaw_aligned[i] or
                    close[i] > ema34_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%