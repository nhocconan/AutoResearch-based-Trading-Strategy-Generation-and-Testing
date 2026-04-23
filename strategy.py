#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d Elder Ray power filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d Elder Ray power > 0 (bullish power) AND 12h volume > 1.8x 20-period MA.
Short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d Elder Ray power < 0 (bearish power) AND 12h volume > 1.8x 20-period MA.
Exit when Williams %R crosses above -20 (for long) or below -80 (for short) or Elder Ray power reverses.
Uses 1d HTF for power filter to align with major trend, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams %R captures mean reversions in bear markets, Elder Ray filters trend alignment.
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
    
    # Calculate Williams %R(14) - momentum oscillator
    williams_r = np.full(n, np.nan)
    for i in range(13, n):  # min_periods=14
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 1d Elder Ray Power (EMA13 close - EMA13 high) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    ema13_close = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_high = pd.Series(high_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    elder_power = ema13_close - ema13_high  # positive = bullish power, negative = bearish power
    elder_power_aligned = align_htf_to_ltf(prices, df_1d, elder_power)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 13, 20)  # Williams %R, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(elder_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossover signals
        wr_cross_above_80 = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_below_20 = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Elder Ray power direction
        power_bullish = elder_power_aligned[i] > 0
        power_bearish = elder_power_aligned[i] < 0
        
        # Volume filter: 12h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND bullish power AND volume filter
            if wr_cross_above_80 and power_bullish and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND bearish power AND volume filter
            elif wr_cross_below_20 and power_bearish and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -20 (overbought) OR power turns bearish
                if williams_r[i] > -20 and williams_r[i-1] <= -20:
                    exit_signal = True
                elif elder_power_aligned[i] < 0:  # power turned bearish
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -80 (oversold) OR power turns bullish
                if williams_r[i] < -80 and williams_r[i-1] >= -80:
                    exit_signal = True
                elif elder_power_aligned[i] > 0:  # power turned bullish
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1dElderRay_Power_VolumeSpike"
timeframe = "12h"
leverage = 1.0