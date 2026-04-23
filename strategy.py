#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike confirmation
- Uses Camarilla pivot levels (R1, S1) from weekly timeframe for breakout signals
- 1w EMA(34) defines trend direction (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 1d timeframe targeting 15-30 trades/year (60-120 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend
- Volume spike requirement reduces false breakouts during low volatility
- Tight entry conditions to minimize fee drag while maintaining edge in BTC/ETH
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
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla pivot levels (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla pivot calculation: based on previous week's OHLC
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + range_1w * 1.1 / 4
    s1 = pivot - range_1w * 1.1 / 4
    
    # Align Camarilla levels to weekly timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R1, uptrend, volume spike
            long_signal = (price_above_r1 and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S1, downtrend, volume spike
            short_signal = (price_below_s1 and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S1 or trend turns down
                if (price_below_s1 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R1 or trend turns up
                if (price_above_r1 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0