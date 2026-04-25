#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d trend filter and ATR-based exits.
Long when Bull Power > 0 and Bear Power < 0 in 1d uptrend (close > 1d EMA50).
Short when Bull Power < 0 and Bear Power > 0 in 1d downtrend (close < 1d EMA50).
Exit when either power crosses zero or ATR trailing stop hit.
Designed for ~12-30 trades/year by requiring strong momentum alignment and trend filter.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via dual confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        
        if position == 0:
            # Only trade in alignment with 1d trend
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: Bull Power > 0 and Bear Power < 0 (strong bullish momentum)
                long_signal = (bull > 0) and (bear < 0)
            else:  # 1d downtrend regime
                # Short: Bull Power < 0 and Bear Power > 0 (strong bearish momentum)
                short_signal = (bull < 0) and (bear > 0)
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: either power crosses zero OR ATR trailing stop
            power_exit = (bull <= 0) or (bear >= 0)
            atr_stop = long_high - 2.5 * atr[i]
            if power_exit or close[i] <= atr_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: either power crosses zero OR ATR trailing stop
            power_exit = (bull >= 0) or (bear <= 0)
            atr_stop = short_low + 2.5 * atr[i]
            if power_exit or close[i] >= atr_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime"
timeframe = "6h"
leverage = 1.0