#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1w trend filter
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when: Alligator aligned bullish (Lips>Teeth>Jaw) AND Bull Power > 0 AND price > 1w EMA(34)
# Short when: Alligator aligned bearish (Lips<Teeth<Jaw) AND Bear Power < 0 AND price < 1w EMA(34)
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following weekly trend.
# Based on proven pattern: Alligator identifies trend, Elder Ray confirms momentum, weekly EMA filters counter-trend.

name = "12h_WilliamsAlligator_ElderRay_1wEMA34_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator components (using median price)
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate Elder Ray components
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any Alligator value is NaN
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR Elder Ray turns negative
            if not (curr_lips > curr_teeth > curr_jaw) or curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR Elder Ray turns positive
            if not (curr_lips < curr_teeth < curr_jaw) or curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment AND Bull Power positive AND price > 1w EMA(34)
            if (curr_lips > curr_teeth > curr_jaw and 
                curr_bull > 0 and 
                curr_close > curr_ema_1w):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment AND Bear Power negative AND price < 1w EMA(34)
            elif (curr_lips < curr_teeth < curr_jaw and 
                  curr_bear < 0 and 
                  curr_close < curr_ema_1w):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals