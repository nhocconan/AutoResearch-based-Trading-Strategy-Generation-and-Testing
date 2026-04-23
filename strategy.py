#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
Target: 12-37 trades/year per symbol. Uses discrete position sizing (0.25) to minimize fee churn.
Williams Alligator (jaw/teeth/lips) identifies trendless markets - only trade when aligned.
Elder Ray (Bull/Bear Power) confirms trend strength. Volume filter avoids false signals.
Works in both bull/bear via trend filter and avoids choppy markets.
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
    
    # Calculate 1d Elder Ray for trend filter (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 12h Williams Alligator (SMAs with specific periods)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Alligator: Jaw (13-period SMA, 8 bars ahead), Teeth (8-period SMA, 5 bars ahead), Lips (5-period SMA, 3 bars ahead)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5) + 8  # need volume MA20, plus Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray trend filter: Bull Power > 0 AND Bear Power < 0 = strong trend
        strong_uptrend = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        strong_downtrend = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA (tight to avoid overtrading)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Alligator aligned up AND Elder Ray uptrend AND volume confirmation
            if alligator_long and strong_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND Elder Ray downtrend AND volume confirmation
            elif alligator_short and strong_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses (teeth crosses lips) OR Elder Ray weakens
            exit_signal = False
            if position == 1:
                # Exit long on Alligator bearish alignment OR Elder Ray turns bearish
                if not alligator_long or not strong_uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short on Alligator bullish alignment OR Elder Ray turns bullish
                if not alligator_short or not strong_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dElderRay_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0