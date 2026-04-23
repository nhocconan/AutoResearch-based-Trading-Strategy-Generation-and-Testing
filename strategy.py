#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d Elder Ray Power filter and volume spike.
Long when Williams %R crosses above -80 (oversold) AND 1d Elder Bull Power > 0 AND volume > 2.0x 20-period MA.
Short when Williams %R crosses below -20 (overbought) AND 1d Elder Bear Power < 0 AND volume > 2.0x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or Elder Power reverses.
Uses 1d HTF for Elder Power trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams %R provides mean reversal signals, Elder Power filters major trend, volume confirms reversal strength.
Works in both bull and bear markets by following the higher timeframe trend via Elder Power.
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
    
    # Calculate 4h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(13, n):  # 14-period lookback
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 1d Elder Ray Power (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Bull Power = High - EMA13
    # Elder Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 13, 20)  # Williams %R, Elder Ray, volume MA
    
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
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Williams %R crossover signals
        wr_cross_up_80 = False
        wr_cross_down_20 = False
        wr_cross_up_50 = False
        wr_cross_down_50 = False
        
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_up_80 = wr_prev <= -80 and wr > -80
            wr_cross_down_20 = wr_prev >= -20 and wr < -20
            wr_cross_up_50 = wr_prev <= -50 and wr > -50
            wr_cross_down_50 = wr_prev >= -50 and wr < -50
        
        # Volume filter: 4h volume > 2.0x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND Bull Power > 0 AND volume filter
            if wr_cross_up_80 and bull_power > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND Bear Power < 0 AND volume filter
            elif wr_cross_down_20 and bear_power < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR Bull Power becomes <= 0
                if wr_cross_up_50 or bull_power <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR Bear Power becomes >= 0
                if wr_cross_down_50 or bear_power >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dElderRay_Power_VolumeSpike"
timeframe = "4h"
leverage = 1.0