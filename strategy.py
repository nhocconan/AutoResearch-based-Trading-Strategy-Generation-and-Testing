#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h Trend Filter
# Williams Alligator identifies trend presence and direction via SMAs (13,8,5).
# Elder Ray measures bull/bear power relative to EMA13 to gauge trend strength.
# Combined with 12h EMA34 trend filter for multi-timeframe confirmation.
# Designed for 12-37 trades/year on 6h timeframe to avoid fee drag.
# Works in trending markets (both bull and bear) by capturing momentum with trend alignment.
name = "6h_WilliamsAlligator_ElderRay_12hTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_6h[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions:
        # Uptrend: Lips > Teeth > Jaw (green alignment)
        # Downtrend: Lips < Teeth < Jaw (red alignment)
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions:
        # Strong bull power: bull_power > 0 and increasing
        # Strong bear power: bear_power < 0 and decreasing
        bull_power_strong = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bear_power_strong = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long: Alligator uptrend + strong bull power + price above 12h EMA34
            if alligator_long and bull_power_strong and close[i] > ema34_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + strong bear power + price below 12h EMA34
            elif alligator_short and bear_power_strong and close[i] < ema34_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down OR bear power becomes strong
            if not alligator_long or bear_power_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up OR bull power becomes strong
            if not alligator_short or bull_power_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals