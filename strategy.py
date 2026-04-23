#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray Power filter and volume spike confirmation.
Williams %R identifies overbought/oversold conditions (long when crosses above -80, short when crosses below -20).
1d Elder Ray Power (Bull Power = High - EMA13, Bear Power = EMA13 - Low) filters trend direction.
Volume spike (>2x 20-period MA) confirms momentum.
Exit when Williams %R crosses opposite threshold or Elder Ray Power reverses.
Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(13, n):  # 14-period needs 13 lookback + current
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 1d Elder Ray Power (EMA13-based) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30, 20)  # Williams %R (needs 13), Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from oversold) AND Bull Power > 0 (uptrend bias) AND volume filter
            if i > start_idx and williams_r[i-1] <= -80 and wr > -80 and bull_power > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from overbought) AND Bear Power > 0 (downtrend bias) AND volume filter
            elif i > start_idx and williams_r[i-1] >= -20 and wr < -20 and bear_power > 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 (momentum loss) OR Bear Power > Bull Power (trend change)
                if i > start_idx and williams_r[i-1] > -50 and wr <= -50:
                    exit_signal = True
                elif bear_power > bull_power:  # Bearish momentum taking over
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 (momentum loss) OR Bull Power > Bear Power (trend change)
                if i > start_idx and williams_r[i-1] < -50 and wr >= -50:
                    exit_signal = True
                elif bull_power > bear_power:  # Bullish momentum taking over
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Power_VolumeSpike"
timeframe = "6h"
leverage = 1.0