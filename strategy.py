#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray regime filter
# - Williams Alligator (13,8,5 SMAs) on 1d for trend direction and alignment
# - Elder Ray (bull/bear power) on 1d to confirm trend strength
# - Entry when price retests Alligator's teeth (8-period SMA) in direction of trend
# - Only trade when Elder Ray shows strong bull/bear power (>0)
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to avoid fee drag
# - Combines trend-following with mean reversion pullbacks for robustness in bull/bear markets

name = "12h_1d_alligator_elder_ray_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Alligator lines (SMAs)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values   # Green line
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    
    # Align Alligator and Elder Ray to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Determine trend direction from Alligator alignment
    # Uptrend: Lips > Teeth > Jaw (green > red > blue)
    # Downtrend: Lips < Teeth < Jaw (green < red < blue)
    uptrend = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    downtrend = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Strong trend confirmation from Elder Ray
    strong_bull = bull_power_aligned > 0
    strong_bear = bear_power_aligned > 0
    
    # Combine for final trend signals
    bullish_trend = uptrend & strong_bull
    bearish_trend = downtrend & strong_bear
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price crosses below teeth OR trend weakness
            if close[i] < teeth_aligned[i]:  # Price retested and failed to hold above teeth
                position = 0
                signals[i] = 0.0
            elif not bullish_trend[i]:  # Trend weakened
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses above teeth OR trend weakness
            if close[i] > teeth_aligned[i]:  # Price retested and failed to hold below teeth
                position = 0
                signals[i] = 0.0
            elif not bearish_trend[i]:  # Trend weakened
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for pullback entries in direction of strong trend
            # Long: price touches or slightly below teeth during uptrend
            if (bullish_trend[i] and 
                low[i] <= teeth_aligned[i] * 1.002 and  # Allow small buffer (0.2%)
                close[i] > teeth_aligned[i]):  # But close above teeth to confirm bounce
                position = 1
                signals[i] = 0.25
            # Short: price touches or slightly above teeth during downtrend
            elif (bearish_trend[i] and 
                  high[i] >= teeth_aligned[i] * 0.998 and  # Allow small buffer (0.2%)
                  close[i] < teeth_aligned[i]):  # But close below teeth to confirm rejection
                position = -1
                signals[i] = -0.25
    
    return signals