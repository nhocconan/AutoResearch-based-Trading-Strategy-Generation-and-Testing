#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h Elder Ray power filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND 12h Bear Power < 0 (bullish bias) AND volume > 2.0x 20-period MA.
Short when Williams %R > -20 (overbought) AND 12h Bull Power < 0 (bearish bias) AND volume > 2.0x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR opposite Elder Ray power turns positive.
Uses 12h HTF for Elder Ray trend filter to avoid counter-trend trades, Williams %R for mean reversion entries, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R provides overextension signals, 12h Elder Ray filters major trend bias, volume confirms reversal strength.
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
        # Use lookback of 14 periods (excluding current bar to avoid look-ahead)
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:  # Avoid division by zero
            williams_r[i] = (hh - close[i]) / (hh - ll) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # Calculate 12h Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13) (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA13
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power_12h = high_12h - ema_13_12h  # Bull Power > 0 = bulls in control
    bear_power_12h = low_12h - ema_13_12h   # Bear Power < 0 = bears in control (more negative = stronger bear)
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 20)  # Williams %R (needs 14), EMA13, volume MA
    
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
        
        # Volume filter: 6h volume > 2.0x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND 12h Bear Power < 0 (bearish bias fading) AND volume filter
            if wr < -80 and bear_power < 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND 12h Bull Power < 0 (bullish bias fading) AND volume filter
            elif wr > -20 and bull_power < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 (leaving oversold) OR 12h Bull Power turns positive (bulls retake control)
                if wr > -50 or bull_power > 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 (leaving overbought) OR 12h Bear Power turns positive (bears lose control)
                if wr < -50 or bear_power > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hElderRay_Power_VolumeSpike"
timeframe = "6h"
leverage = 1.0