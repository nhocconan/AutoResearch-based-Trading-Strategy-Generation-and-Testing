#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot S1/R1 breakout with 1d trend filter and volume confirmation.
Long when price breaks above S1 with bullish 1d EMA trend and volume spike.
Short when price breaks below R1 with bearish 1d EMA trend and volume spike.
Exit when price crosses the 12h EMA34 or trend weakens.
Designed for low trade frequency (12-25/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 40:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for exit
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema34_d = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate 12h Camarilla levels from previous day
    # Camarilla: H/L/C from prior day
    # S1 = C - (H-L)*1.0833/2
    # R1 = C + (H-L)*1.0833/2
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    camarilla_range = (prev_high - prev_low) * 1.0833 / 2.0
    s1_d = prev_close - camarilla_range
    r1_d = prev_close + camarilla_range
    
    # Align Camarilla levels to 12h (previous day's levels valid until next day)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_d)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(ema34[i]) or np.isnan(ema34_d_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
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
            # Long: Price breaks above S1 with bullish 1d trend and volume
            if (close[i] > s1_aligned[i] and 
                ema34_d_aligned[i] > ema34_d_aligned[max(i-1,0)] and  # Rising trend
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with bearish 1d trend and volume
            elif (close[i] < r1_aligned[i] and 
                  ema34_d_aligned[i] < ema34_d_aligned[max(i-1,0)] and  # Falling trend
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below EMA34 OR trend turns bearish
                if close[i] < ema34[i] or ema34_d_aligned[i] < ema34_d_aligned[max(i-1,0)]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above EMA34 OR trend turns bullish
                if close[i] > ema34[i] or ema34_d_aligned[i] > ema34_d_aligned[max(i-1,0)]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_S1R1_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%