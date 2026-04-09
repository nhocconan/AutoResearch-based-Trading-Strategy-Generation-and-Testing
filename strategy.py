#!/usr/bin/env python3
# 6h_elder_ray_alligator_regime_v1
# Hypothesis: 6h strategy combining Elder Ray (Bull/Bear Power) with Williams Alligator for trend confirmation and regime filtering.
# Uses 1d HTF EMA(50) for higher timeframe alignment. Long when Bull Power > 0, price above Alligator teeth, and EMA50 uptrend.
# Short when Bear Power < 0, price below Alligator teeth, and EMA50 downtrend.
# Designed to work in both bull (trend following) and bear (counter-trend retracements) markets via regime adaptation.
# Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_alligator_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) - all SMMA
    def smma(src, length):
        # Smoothed Moving Average: first value is SMA, then recursive
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < length:
            return result
        # First value: simple moving average
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_SRC) / length
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)   # Red line
    lips = smma(close, 5)    # Green line
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup (max of 13, 50)
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(close[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Teeth above/below Jaw indicates trend direction
        # In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw
        # We use Teeth vs Jaw for trend, Lips for momentum confirmation
        teeth_above_jaw = teeth[i] > jaw[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bullish_pressure = bull_power[i] > 0
        bearish_pressure = bear_power[i] < 0
        
        # HTF trend filter
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative OR price closes below Teeth
            if bear_power[i] < 0 or close[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR price closes above Teeth
            if bull_power[i] > 0 or close[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power positive, price above Teeth, Teeth above Jaw, HTF uptrend
            bullish_entry = (bullish_pressure and 
                           close[i] > teeth[i] and 
                           teeth_above_jaw and 
                           htf_uptrend)
                           
            # Enter short: Bear Power negative, price below Teeth, Teeth below Jaw, HTF downtrend
            bearish_entry = (bearish_pressure and 
                           close[i] < teeth[i] and 
                           teeth_below_jaw and 
                           htf_downtrend)
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals