#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combination with 1d trend filter.
# Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Williams Alligator (13,8,5 SMAs) provides trend direction and entry signals.
# Elder Ray (Bull/Bear Power from 1d EMA13) confirms trend strength.
# 1d EMA34 acts as higher timeframe trend filter: only trade in direction of 1d trend.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "12h_WilliamsAlligator_ElderRay_1dEMA34_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Williams Alligator and 1d data for Elder Ray/1d trend
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 12h: Jaw(13), Teeth(8), Lips(5)
    # All SMAs with proper min_periods
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d  # Negative values indicate bearish pressure
    
    # 1d EMA34 for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions:
        # Lips > Teeth > Jaw = bullish alignment (green)
        # Lips < Teeth < Jaw = bearish alignment (red)
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Elder Ray conditions:
        # Bull Power > 0 = bullish pressure
        # Bear Power < 0 = bearish pressure (more negative = stronger bearish)
        bullish_elder = bull_power_aligned[i] > 0
        bearish_elder = bear_power_aligned[i] < 0
        
        # 1d EMA34 trend filter:
        # Price above EMA34 = bullish trend
        # Price below EMA34 = bearish trend
        price_above_ema34 = close[i] > ema_34_1d_aligned[i]
        price_below_ema34 = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions require ALL three to agree:
        # 1. Alligator alignment
        # 2. Elder Ray confirmation
        # 3. 1d trend filter
        long_entry = bullish_alligator and bullish_elder and price_above_ema34
        short_entry = bearish_alligator and bearish_elder and price_below_ema34
        
        # Exit conditions: opposite Alligator alignment (trend change)
        long_exit = bearish_alligator  # Exit long when Alligator turns bearish
        short_exit = bullish_alligator  # Exit short when Alligator turns bullish
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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