#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator (13,8,5 SMAs) with 1d trend filter and volume confirmation.
Long when green Alligator (JAW>TEETH>LIPS) with bullish 1d trend and volume spike.
Short when red Alligator (LIPS>TEETH>JAW) with bearish 1d trend and volume spike.
Exit when Alligator turns neutral (intertwined) or trend weakens.
Designed for low trade frequency (20-40/year) to minimize fee drag.
Alligator indicator works well in trending markets and avoids whipsaws in ranges.
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
    
    # Align EMA34 to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate Williams Alligator on 4h timeframe
    # Jaw: 13-period SMMA (smoothed with 8-period offset)
    # Teeth: 8-period SMMA (smoothed with 5-period offset)
    # Lips: 5-period SMMA (smoothed with 3-period offset)
    
    # SMMA calculation (Smoothed Moving Average)
    def smma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Alligator lookback
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
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
            # Long: Green Alligator (JAW > TEETH > LIPS) with bullish 1d trend and volume spike
            if (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Green Alligator
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Red Alligator (LIPS > TEETH > JAW) with bearish 1d trend and volume spike
            elif (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Red Alligator
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns neutral or trend turns bearish
                if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator turns neutral or trend turns bullish
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] > ema34_aligned[i]:
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