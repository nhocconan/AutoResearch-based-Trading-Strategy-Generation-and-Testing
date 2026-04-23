#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA200 trend filter and volume confirmation
- Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
- Extreme readings (> -10 for long, < -90 for short) signal potential reversals
- Trade only in direction of 1d EMA(200) to avoid counter-trend whipsaws in bear markets
- Volume confirmation (> 2.0x 20-period average) ensures reversal has momentum
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading reversals with the 1d trend
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
    
    # Get 1d data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (extra delay needed for indicator confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r, additional_delay_bars=0)
    
    # Calculate 1d EMA(200) for trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme conditions
        williams_r_val = williams_r_aligned[i]
        oversold_extreme = williams_r_val < -90  # Deep oversold
        overbought_extreme = williams_r_val > -10  # Deep overbought
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long conditions: oversold extreme, uptrend, volume spike
            long_signal = (oversold_extreme and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: overbought extreme, downtrend, volume spike
            short_signal = (overbought_extreme and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or trend turns down
                if (williams_r_val > -50 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 or trend turns up
                if (williams_r_val < -50 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0