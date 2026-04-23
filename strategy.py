#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 Breakout with 1d EMA34 Trend Filter and Volume Spike + Chop Regime Filter
- Uses Camarilla pivot levels (R1/S1) from daily timeframe for tighter structure-based entries
- 1d EMA34 defines trend filter: only trade in direction of daily trend
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Chop regime filter: only trade when market is trending (CHOP < 38.2) to avoid whipsaws
- Designed for 4h timeframe targeting 30-60 trades/year (120-240 over 4 years)
- Works in both bull and bear markets by combining mean reversion at extremes with trend filter and regime filter
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
    
    # Calculate daily Camarilla pivot levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Shift by 1 to use previous day's OHLC for today's levels (no look-ahead)
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    # First value will be NaN due to roll, which is correct (no previous day)
    
    # Camarilla calculations: based on previous day's range
    PP = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    R = high_1d_prev - low_1d_prev
    R1 = PP + R * 1.1 / 12
    S1 = PP - R * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (trending when CHOP < 38.2)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_period = 14
    chop_period = 14
    tr = np.maximum(high - low, np.absolute(np.roll(high, 1) - close), np.absolute(np.roll(low, 1) - close))
    # Handle first bar TR
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(chop_period)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)  # Default to neutral when range=0
    chop = np.where(np.isnan(chop), 50.0, chop)  # Handle NaN
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, chop_period)  # for volume MA, EMA34, and chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 1d EMA34 AND volume spike AND trending market (CHOP < 38.2)
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 1d EMA34 AND volume spike AND trending market (CHOP < 38.2)
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retouches PP OR trend reverses OR chop becomes too high (range-bound)
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches PP OR closes below 1d EMA34 OR chop too high
                if (close[i] <= PP_aligned[i] or close[i] < ema_34_1d_aligned[i] or chop[i] > 61.8):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches PP OR closes above 1d EMA34 OR chop too high
                if (close[i] >= PP_aligned[i] or close[i] > ema_34_1d_aligned[i] or chop[i] > 61.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeChop"
timeframe = "4h"
leverage = 1.0