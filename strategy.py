#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA(34) trend filter and volume confirmation (>1.3x 20-period average)
- Camarilla pivot levels (R1/S1) from 1d timeframe provide high-probability intraday support/resistance
- 12h timeframe balances trade frequency and signal quality for 12-37 trades/year target
- Volume confirmation validates breakout strength to avoid false signals
- Works in both bull and bear markets by trading with 1d trend direction from daily EMA
- Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 1d bar
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + (1.1 * camarilla_range / 12)
    s1 = prev_close - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 12h timeframe (extra delay not needed as based on completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        # Long: price breaks above R1 level
        # Short: price breaks below S1 level
        long_breakout = close[i] > r1_aligned[i]
        short_breakout = close[i] < s1_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above R1, uptrend, volume spike
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 1.3 * vol_ma[i])
            
            # Short conditions: breakout below S1, downtrend, volume spike
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 1.3 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Camarilla level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S1 level or trend turns down
                if (close[i] < s1_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R1 level or trend turns up
                if (close[i] > r1_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0