#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for structure and lower fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Entry: Long when price breaks above Camarilla R1 AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below Camarilla S1 AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level (S1 for long, R1 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This combines price channel breakouts with trend filtering and volume confirmation to capture
institutional moves while avoiding choppy markets. Works in both bull and bear markets by
only taking trades in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 4h bar
    # R1 = close + 1.1/12 * (high - low)
    # S1 = close - 1.1/12 * (high - low)
    # Using previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + (1.1 / 12) * camarilla_range
    s1 = prev_close - (1.1 / 12) * camarilla_range
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough bars for EMA34, volume MA, and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: break above R1 AND 1d EMA34 bullish (close > EMA)
                if curr_high > r1[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: break below S1 AND 1d EMA34 bearish (close < EMA)
                elif curr_low < s1[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: break below S1 OR loss of volume confirmation
            if curr_low < s1[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R1 OR loss of volume confirmation
            if curr_high > r1[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0