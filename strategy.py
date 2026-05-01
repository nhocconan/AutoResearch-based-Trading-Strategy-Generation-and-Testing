#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter.
# Uses 1d EMA34 for trend direction, 6h Alligator (jaw/teeth/lips) for trend strength,
# and 6h Elder Ray to measure bull/bear power relative to 13-period EMA.
# Long when: price > Alligator lips, Bull Power > 0, and 1d EMA34 uptrend.
# Short when: price < Alligator lips, Bear Power < 0, and 1d EMA34 downtrend.
# Discrete position sizing (0.25) to manage drawdown. Target: 50-150 trades over 4 years.

name = "6h_Alligator_ElderRay_1dEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # Using regular SMA with min_periods for simplicity (close to SMMA)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Calculate 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (max shift is 8 for jaw)
    start_idx = 13 + 8  # 21 bars for jaw calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw.iloc[i]) or
            np.isnan(teeth.iloc[i]) or
            np.isnan(lips.iloc[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1d EMA34 direction (using previous bar to avoid look-ahead)
        uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, reverse = downtrend
        alligator_long = lips.iloc[i] > teeth.iloc[i] > jaw.iloc[i]
        alligator_short = lips.iloc[i] < teeth.iloc[i] < jaw.iloc[i]
        
        # Elder Ray: Bull Power > 0 = bullish pressure, Bear Power < 0 = bearish pressure
        bull_pressure = bull_power[i] > 0
        bear_pressure = bear_power[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend AND bull power positive AND 1d uptrend
            if alligator_long and bull_pressure and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND bear power negative AND 1d downtrend
            elif alligator_short and bear_pressure and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator death cross (lips < teeth) or bear power negative
            if lips.iloc[i] < teeth.iloc[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator golden cross (lips > teeth) or bull power positive
            if lips.iloc[i] > teeth.iloc[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals