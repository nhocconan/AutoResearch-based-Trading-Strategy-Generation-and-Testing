#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter, volume confirmation, and session filter (08-20 UTC)
- Long when price breaks above Camarilla R1 level AND close > 4h EMA50 AND volume > 1.8x 20-period average AND in session
- Short when price breaks below Camarilla S1 level AND close < 4h EMA50 AND volume > 1.8x 20-period average AND in session
- Exit when price crosses Camarilla pivot point (mean reversion to center)
- Uses 4h EMA50 for HTF trend alignment to avoid counter-trend entries
- Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
- Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
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
    
    # Get 4h data for EMA50 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        in_session = 8 <= hours[i] <= 20
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below S1
        
        # Trend filter (using 4h EMA50)
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + uptrend + volume confirmation + in session
            if breakout_up and uptrend and volume_ok and in_session:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down + downtrend + volume confirmation + in session
            elif breakout_down and downtrend and volume_ok and in_session:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price crosses Camarilla pivot point (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below pivot point
                if close[i] < camarilla_pp_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above pivot point
                if close[i] > camarilla_pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hEMA50_Trend_VolumeConfirmation_SessionFilter"
timeframe = "1h"
leverage = 1.0