#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator system with 1d EMA50 trend filter and volume confirmation.
Long when price is above alligator jaws (blue line) with bullish 1d trend and volume spike.
Short when price is below alligator jaws with bearish 1d trend and volume spike.
Exit when price crosses back below/above jaws or trend weakens.
Uses Williams Alligator (SMAs: 13,8,5) to identify trends and avoid whipsaws.
Designed for low trade frequency (20-40/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Alligator and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (SMAs of median price)
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    median_price = (high + low) / 2.0
    
    # Calculate SMAs for Alligator lines
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Jaws (13-period SMA)
    jaws_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 4h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_aligned[i]) or 
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
            # Long: Price above jaws with bullish 1d trend and volume spike
            # Alligator alignment: Lips > Teeth > Jaws (bullish alignment)
            if (close[i] > jaws_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaws_aligned[i] and
                close[i] > ema50_aligned[i] and  # Price above EMA50 for trend
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price below jaws with bearish 1d trend and volume spike
            # Alligator alignment: Lips < Teeth < Jaws (bearish alignment)
            elif (close[i] < jaws_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaws_aligned[i] and
                  close[i] < ema50_aligned[i] and  # Price below EMA50 for trend
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below jaws OR alligator alignment turns bearish
                if close[i] < jaws_aligned[i] or not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above jaws OR alligator alignment turns bullish
                if close[i] > jaws_aligned[i] or not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%