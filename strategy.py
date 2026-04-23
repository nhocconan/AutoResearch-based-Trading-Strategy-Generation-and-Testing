#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray (Bull/Bear Power) trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND Bear Power > 0 (bullish bias) AND volume > 1.5x 20-period MA.
Short when Williams %R > -20 (overbought) AND Bull Power < 0 (bearish bias) AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or trend filter reverses.
Uses 1d HTF for Elder Ray to capture major trend bias, Williams %R for precise 6h entry timing, volume for momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R provides mean-reversion signals in ranging markets, Elder Ray filters for trend alignment,
volume confirms breakout strength. Works in both bull and bear markets by following higher timeframe trend bias.
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
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 1d Elder Ray (Bull/Bear Power) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # EMA13 needs min_periods
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 of close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 20)  # Williams %R (needs 14), Elder Ray (needs 13), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND Bear Power > 0 (bullish bias) AND volume filter
            if wr < -80 and bear_val > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND Bull Power < 0 (bearish bias) AND volume filter
            elif wr > -20 and bull_val < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR Bear Power becomes negative (trend change)
                if wr > -50 or (i >= start_idx + 1 and bear_val < 0 and bear_power_aligned[i-1] >= 0):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR Bull Power becomes positive (trend change)
                if wr < -50 or (i >= start_idx + 1 and bull_val > 0 and bull_power_aligned[i-1] <= 0):
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