#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(15) breakout with 1d trend filter and volume confirmation.
Long when price breaks above upper band with bullish 1d trend and volume spike.
Short when price breaks below lower band with bearish 1d trend and volume spike.
Exit when price crosses opposite band or trend reverses.
Designed for low trade frequency (15-30/year) to minimize fee drift.
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
    
    # Load 1d data for trend filter and Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian bands on 12h using 15-period
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    
    # Upper band: highest high of last 15 periods
    upper_band = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    # Lower band: lowest low of last 15 periods
    lower_band = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Align bands to 12h timeframe (but we're already on 12h, so just align)
    upper_band_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), lower_band)
    
    # Calculate 12h volume average (20-period)
    vol_12h = get_htf_data(prices, '12h')['volume'].values
    vol_avg_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
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
            # Long: Price breaks above upper band with bullish 1d trend and volume spike
            if (close[i] > upper_band_aligned[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 1.8 * vol_avg_aligned[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with bearish 1d trend and volume spike
            elif (close[i] < lower_band_aligned[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 1.8 * vol_avg_aligned[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below lower band or trend turns bearish
                if (close[i] < lower_band_aligned[i] or 
                    close[i] < ema50_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above upper band or trend turns bullish
                if (close[i] > upper_band_aligned[i] or 
                    close[i] > ema50_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_15_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0