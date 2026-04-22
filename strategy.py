#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Point Breakout with 1-day EMA trend filter and volume spike confirmation.
Enter long when price breaks above R1 level with bullish 1-day EMA trend and above-average volume.
Enter short when price breaks below S1 level with bearish 1-day EMA trend and above-average volume.
Exit when price crosses back below/above pivot point (PP).
Camarilla levels provide precise intraday support/resistance, EMA filter ensures trend alignment,
and volume spike confirms institutional participation. Works in bull/bear markets by following
institutional volume while using pivot levels for precise entry/exit.
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
    
    # Load 1-day data for EMA trend and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1-day average volume for volume spike filter
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Typical Camarilla formula: 
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    
    # We need previous day's data to calculate today's levels
    # Shift the 1-day data by 1 to get previous day's OHLC
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current day based on previous day
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.0833)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.0833)
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, bullish EMA trend, volume spike
            if (close[i] > r1_aligned[i] and 
                ema_34_aligned[i] > ema_34_aligned[i-1] and  # EMA rising
                volume[i] > avg_vol_1d_aligned[i]):          # Volume above average
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, bearish EMA trend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema_34_aligned[i] < ema_34_aligned[i-1] and  # EMA falling
                  volume[i] > avg_vol_1d_aligned[i]):          # Volume above average
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below pivot point
                if close[i] < pp_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above pivot point
                if close[i] > pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0