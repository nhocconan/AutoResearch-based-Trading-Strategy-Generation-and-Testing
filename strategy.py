#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
# Enter long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA),
# 1d Elder Ray bull power > 0, and volume > 1.5x 20-bar average.
# Enter short when jaws cross below teeth, 1d Elder Ray bear power < 0, and volume confirmation.
# Exit when Alligator jaws cross back in opposite direction or Elder Ray power reverses.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
# Target: 50-120 total trades over 4 years (12-30/year) to avoid excessive fee churn.
# Williams Alligator identifies trend initiation; Elder Ray confirms bull/bear power from higher timeframe;
# Volume spike validates institutional participation. Works in both bull (trend following) and bear (counter-trend retracements).

name = "4h_Williams_Alligator_1dElderRay_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of Median Price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    median_price = (high + low) / 2
    
    def smma(data, period):
        """Calculate Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (blue)
    teeth = smma(median_price, 8)  # Teeth (red)
    lips = smma(median_price, 5)   # Lips (green) - not used in signals but helps visualization
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Alligator conditions: Jaw cross Teeth
        jaw_above_teeth = jaw[i] > teeth[i]
        jaw_below_teeth = jaw[i] < teeth[i]
        jaw_above_teeth_prev = jaw[i-1] > teeth[i-1] if i > 0 else False
        jaw_below_teeth_prev = jaw[i-1] < teeth[i-1] if i > 0 else False
        
        # Bullish cross: Jaw crosses above Teeth
        bullish_cross = jaw_above_teeth and not jaw_above_teeth_prev
        # Bearish cross: Jaw crosses below Teeth
        bearish_cross = jaw_below_teeth and not jaw_below_teeth_prev
        
        # Elder Ray conditions from 1d
        bull_power_pos = bull_power_aligned[i] > 0
        bear_power_neg = bear_power_aligned[i] < 0
        
        # Exit conditions: Alligator cross reverses or Elder Ray power reverses
        exit_long = (jaw_below_teeth and not jaw_below_teeth_prev) or bear_power_neg
        exit_short = (jaw_above_teeth and not jaw_above_teeth_prev) or bull_power_pos
        
        # Handle entries and exits
        if bullish_cross and bull_power_pos and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif bearish_cross and bear_power_neg and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals