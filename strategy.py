#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
- Uses Camarilla pivot levels (R3, S3) from 1d for high-probability breakout entries
- 1w EMA(34) defines primary trend: long only when price > EMA34, short only when price < EMA34
- Volume confirmation (> 1.5x 20-period average) ensures breakout has momentum
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in bull markets via trend-following breakouts and bear markets via mean-reversion at extremes
- Camarilla R3/S3 levels represent strong support/resistance where breakouts often trend
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
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation: based on previous day's range
    # R4 = close + (high-low)*1.5/2, R3 = close + (high-low)*1.25/2, etc.
    # We use R3 and S3 as the key levels
    diff_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + diff_1d * 1.25 / 2
    camarilla_s3_1d = close_1d - diff_1d * 1.25 / 2
    
    # Align Camarilla levels to 1d timeframe (already aligned as daily values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 1w EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(2, 34, 20)  # Camarilla needs 2 days, EMA34 needs 34 weeks, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r3 = close[i] > camarilla_r3_aligned[i]
        price_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: price > EMA34 for longs, price < EMA34 for shorts
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R3, in uptrend, volume spike
            long_signal = (price_above_r3 and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3, in downtrend, volume spike
            short_signal = (price_below_s3 and 
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
                # Exit long: price falls below S3 or trend turns down
                if (price_below_s3 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend turns up
                if (price_above_r3 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0