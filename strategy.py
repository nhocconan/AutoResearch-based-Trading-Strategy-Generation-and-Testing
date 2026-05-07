#!/usr/bin/env python3
name = "6h_WilliamsAlligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once for trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5) SMAs on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align Alligator lines to 6h
    jaw_aligned = align_ltf_to_htf(prices, df_1d, jaw)
    teeth_aligned = align_ltf_to_htf(prices, df_1d, teeth)
    lips_aligned = align_ltf_to_htf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: Alligator bullish + Elder Ray bull + price above lips + 1d uptrend
            if alligator_bullish and bull_strong and close[i] > lips_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Elder Ray bear + price below lips + 1d downtrend
            elif alligator_bearish and bear_strong and close[i] < lips_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator turns bearish or Elder Ray turns negative
            if not alligator_bullish or not bull_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator turns bullish or Elder Ray turns positive
            if not alligator_bearish or not bear_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator identifies trend structure, Elder Ray confirms momentum,
# 1d EMA34 filter ensures higher timeframe trend alignment. Works in both bull/bear markets
# by following the Alligator's alignment. Volume filter reduces false signals.
# Williams Alligator (13,8,5) defines market structure: when aligned (lips-teeth-jaw in order),
# it indicates a strong trend. Elder Ray (Bull/Bear Power) measures buying/selling pressure
# relative to EMA13. The 1d EMA34 filter ensures we only take trades in the direction of
# the daily trend. This combination avoids whipsaws and captures sustained moves.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within limits.
# Position size 0.25 balances return and drawdown control.