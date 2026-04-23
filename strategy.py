#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray (Bull/Bear Power) regime filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold reversal) AND 1d Bull Power > 0 AND volume > 1.2x 20-period average.
Short when Williams %R crosses below -20 (overbought reversal) AND 1d Bear Power < 0 AND volume > 1.2x 20-period average.
Exit when Williams %R crosses opposite threshold (-50 for long exit, -50 for short exit) or regime changes.
Uses 1d HTF for regime (Elder Power) to ensure alignment with higher timeframe momentum, reducing false reversals in chop.
Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe.
Williams %R captures short-term exhaustion, Elder Ray filters for genuine momentum, volume avoids low-conviction signals.
Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
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
    
    # Calculate 1d Elder Ray for regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h Williams %R (14-period)
    lookback = 14
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 6h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 13, 20)  # Williams %R, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Volume filter: 6h volume > 1.2x 20-period average
        vol_filter = volume[i] > 1.2 * vol_ma_val
        
        # Williams %R thresholds
        wr_oversold = -80
        wr_overbought = -20
        wr_exit = -50
        
        # Williams %R crossover signals
        if i > start_idx:
            wr_prev = williams_r[i-1]
            wr_cross_above_oversold = wr_prev <= wr_oversold and wr > wr_oversold
            wr_cross_below_overbought = wr_prev >= wr_overbought and wr < wr_overbought
            wr_cross_above_exit = wr_prev <= wr_exit and wr > wr_exit
            wr_cross_below_exit = wr_prev >= wr_exit and wr < wr_exit
        else:
            wr_cross_above_oversold = False
            wr_cross_below_overbought = False
            wr_cross_above_exit = False
            wr_cross_below_exit = False
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND Bull Power > 0 AND volume filter
            if wr_cross_above_oversold and bull_power > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND Bear Power < 0 AND volume filter
            elif wr_cross_below_overbought and bear_power < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR Bull Power becomes negative
                if wr_cross_above_exit or bull_power <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR Bear Power becomes positive
                if wr_cross_below_exit or bear_power >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Regime_VolumeFilter"
timeframe = "6h"
leverage = 1.0