#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray and volume confirmation.
# Williams Alligator identifies trend direction using smoothed medians.
# Elder Ray measures bull/bear power behind the trend.
# Volume confirmation ensures momentum behind moves.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5) on close
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Smoothed medians (using close)
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).median().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).median().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).median().values
    
    # Shift to avoid look-ahead (Alligator uses future data in calculation)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align weekly trend filter
    close_1w = df_1w['close'].values
    sma_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Average volume (20-period) for confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(sma_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_align = lips[i] > teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray confirmation
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0,i-9):i+1])
        strong_bear = bear_power[i] < 0 and bear_power[i] < np.mean(bear_power[max(0,i-9):i+1])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * avg_volume[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma_1w_aligned[i]
        weekly_downtrend = close[i] < sma_1w_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish + Elder Ray bull + volume + weekly uptrend
            if (bullish_align and strong_bull and vol_confirm and weekly_uptrend):
                position = 1
                signals[i] = position_size
            # Short: Alligator bearish + Elder Ray bear + volume + weekly downtrend
            elif (bearish_align and strong_bear and vol_confirm and weekly_downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR Elder Ray turns negative
            if not bullish_align or bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish OR Elder Ray turns positive
            if not bearish_align or bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0