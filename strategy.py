#!/usr/bin/env python3
"""
Hypothesis: 1d weekly pivot (R1/S1) breakout with 1w trend filter and volume confirmation.
Long when price breaks above weekly R1 with bullish weekly trend and volume spike.
Short when price breaks below weekly S1 with bearish weekly trend and volume spike.
Exit when price crosses weekly pivot (PP) or trend weakens.
Uses weekly trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (10-25/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema34_weekly = pd.Series(df_weekly['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly pivot points (PP, R1, S1) from prior week
    # PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp_weekly = (weekly_high + weekly_low + weekly_close) / 3.0
    r1_weekly = 2 * pp_weekly - weekly_low
    s1_weekly = 2 * pp_weekly - weekly_high
    
    # Align weekly data to daily timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Wait for EMA34 to be valid
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with bullish weekly trend and volume
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Price above weekly EMA34 = bullish trend
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with bearish weekly trend and volume
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Price below weekly EMA34 = bearish trend
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below weekly PP OR price below weekly EMA34
                if close[i] < pp_aligned[i] or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above weekly PP OR price above weekly EMA34
                if close[i] > pp_aligned[i] or close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyPivot_R1S1_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0
#%%