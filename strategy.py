#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1w/1d Regime Filter.
Long when Jaw < Teeth < Lips (bullish alignment) and 1w ADX > 25 (trending up).
Short when Jaw > Teeth > Lips (bearish alignment) and 1w ADX > 25 (trending down).
Exit when alignment breaks or ADX < 20 (range regime).
Uses 1w for ADX regime, 12h for Alligator (SMAs based on Williams Alligator).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1w ADX
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate 12h Williams Alligator
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(values, period):
        """Smoothed Moving Average (Williams Alligator)"""
        sma = np.zeros_like(values)
        sma[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        return sma
    
    # Calculate SMMA for different periods
    smma13 = smma(close, 13)
    smma8 = smma(close, 8)
    smma5 = smma(close, 5)
    
    # Apply shifts (Williams Alligator specific)
    jaw = np.roll(smma13, 8)   # Jaw: 13-period SMMA shifted 8 bars
    teeth = np.roll(smma8, 5)  # Teeth: 8-period SMMA shifted 5 bars
    lips = np.roll(smma5, 3)   # Lips: 5-period SMMA shifted 3 bars
    
    # Fill NaN values from roll with initial values
    jaw[:8] = jaw[8] if len(jaw) > 8 else 0
    teeth[:5] = teeth[5] if len(teeth) > 5 else 0
    lips[:3] = lips[3] if len(lips) > 3 else 0
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_14_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime determination from 1w ADX
        adx_val = adx_14_1w_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Williams Alligator signals
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_alignment = jaw_val < teeth_val and teeth_val < lips_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Long: Bullish alignment AND trending regime
            if bullish_alignment and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND trending regime
            elif bearish_alignment and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bullish alignment breaks OR regime becomes ranging
            if not bullish_alignment or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bearish alignment breaks OR regime becomes ranging
            if not bearish_alignment or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wADX_Regime"
timeframe = "12h"
leverage = 1.0