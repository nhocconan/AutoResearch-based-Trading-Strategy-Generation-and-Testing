#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 12h Camarilla R3 (resistance level) AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 12h Camarilla S3 (support level) AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 12h Camarilla midpoint (mean reversion to median)
- Uses 1d EMA34 for trend alignment to avoid counter-trend trades and capture major trend
- Volume spike ensures institutional participation and reduces false breakouts
- Camarilla pivot levels provide high-probability reversal points in ranging markets
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
- Primary timeframe: 12h (slower timeframe reduces overtrading risk)
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
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Camarilla formulas: based on previous day's high, low, close
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    # We'll use R3 and S3 as breakout levels, and PP as midpoint for exit
    prev_high = df_12h['high'].shift(1).values  # Previous 12h bar high
    prev_low = df_12h['low'].shift(1).values    # Previous 12h bar low
    prev_close = df_12h['close'].shift(1).values # Previous 12h bar close
    
    # Calculate pivot points
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4.0)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4.0)
    
    # Align 12h Camarilla levels to 12h LTF
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 21)  # Need 35 for EMA34 (34+1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r3_aligned[i]  # Break above Camarilla R3
        breakout_down = close[i] < s3_aligned[i]  # Break below Camarilla S3
        
        # Trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Camarilla midpoint (PP)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint (PP)
                if close[i] < pp_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint (PP)
                if close[i] > pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0