#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray (Bull/Bear Power) filter
# Combines Williams Alligator (JAWS/TEETH/LIPS) for trend direction and Elder Ray
# for momentum strength. Long when Lips > Teeth > Jaws AND Bull Power > 0 with rising trend.
# Short when Lips < Teeth < Jaws AND Bear Power < 0 with falling trend.
# Uses weekly trend filter to avoid counter-trend trades in strong monthly trends.
# Designed for 6h timeframe: 12-35 trades/year (~50-140 total over 4 years).
# Works in bull markets via trend following and in bear markets via short signals
# when Alligator is bearish and Bear Power confirms selling pressure.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load higher timeframe data: weekly for trend filter, daily for Alligator and Elder Ray
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily timeframe (13,8,5 SMAs shifted)
    close_d = df_1d['close'].values
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    
    # Jaw (13-period SMMA, shifted 8 bars forward)
    jaw = pd.Series(close_d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, shifted 5 bars forward)
    teeth = pd.Series(close_d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, shifted 3 bars forward)
    lips = pd.Series(close_d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13_d = pd.Series(close_d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_d - ema13_d
    bear_power = low_d - ema13_d
    
    # Weekly trend filter: price vs 40-week EMA
    close_w = df_1w['close'].values
    ema40_w = pd.Series(close_w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align all indicators to 6h timeframe (wait for daily/weekly close)
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema40_w_6h = align_htf_to_ltf(prices, df_1w, ema40_w)
    
    # Pre-compute session hours (08-20 UTC) for liquidity
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or np.isnan(ema40_w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        
        # Williams Alligator conditions
        lips_above_teeth = lips_6h[i] > teeth_6h[i]
        teeth_above_jaw = teeth_6h[i] > jaw_6h[i]
        lips_below_teeth = lips_6h[i] < teeth_6h[i]
        teeth_below_jaw = teeth_6h[i] < jaw_6h[i]
        
        # Elder Ray conditions
        bull_power_pos = bull_power_6h[i] > 0
        bear_power_neg = bear_power_6h[i] < 0
        
        # Weekly trend filter
        price_above_weekly_ema = price_close > ema40_w_6h[i]
        price_below_weekly_ema = price_close < ema40_w_6h[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaws (bullish alignment) AND Bull Power > 0 AND price above weekly EMA
            if (lips_above_teeth and teeth_above_jaw and bull_power_pos and price_above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaws (bearish alignment) AND Bear Power < 0 AND price below weekly EMA
            elif (lips_below_teeth and teeth_below_jaw and bear_power_neg and price_below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR Bull Power turns negative
                if not (lips_above_teeth and teeth_above_jaw) or bull_power_6h[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns bullish OR Bear Power turns positive
                if not (lips_below_teeth and teeth_below_jaw) or bear_power_6h[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dElderRay_1wTrendFilter"
timeframe = "6h"
leverage = 1.0