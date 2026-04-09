#!/usr/bin/env python3
# 4h_1w_volatility_breakout_v1
# Hypothesis: 4h strategy using weekly volatility breakout with volume confirmation and ATR trailing stop.
# Long: Price breaks above weekly high + 1.5*ATR(1w) with volume > 1.3x 20-period average.
# Short: Price breaks below weekly low - 1.5*ATR(1w) with volume > 1.3x 20-period average.
# Exit: ATR trailing stop (2.0x ATR from extreme) or opposite weekly breakout.
# Uses weekly volatility expansion for major moves, 4h for execution, volume for confirmation, ATR for dynamic stops.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for volatility breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    tr1_1w = high_1w - low_1w
    tr2_1w = (high_1w - close_1w.shift()).abs()
    tr3_1w = (low_1w - close_1w.shift()).abs()
    tr_1w = pd.concat([tr1_1w, tr2_1w, tr3_1w], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly high/low for breakout levels
    weekly_high = high_1w.rolling(window=20, min_periods=20).max().values
    weekly_low = low_1w.rolling(window=20, min_periods=20).min().values
    
    # Calculate breakout levels: weekly extreme ± 1.5*weekly ATR
    breakout_up = weekly_high + 1.5 * atr_1w
    breakout_down = weekly_low - 1.5 * atr_1w
    
    # Align HTF breakout levels to 4h timeframe (wait for completed 1w bar)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below weekly breakout down level
            elif close[i] < breakout_down_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above weekly breakout up level
            elif close[i] > breakout_up_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation
            bullish_breakout = (close[i] > breakout_up_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < breakout_down_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals