#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 1d EMA50
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 1d EMA50
# Uses 13,8,5 SMAs for Alligator and 13-period EMA for Elder Ray power calculation
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Alligator identifies trend, Elder Ray measures bull/bear power behind the move, 1d EMA50 filters counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Williams Alligator (13,8,5) ===
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMMA (approximated by SMA)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMMA
    
    # === 6h Elder Ray Power (Bull/Bear) ===
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d EMA50 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        ema50_val = ema50_1d_aligned[i]
        
        # === EXIT LOGIC (Alligator reversal) ===
        if position == 1:  # Long position
            # Exit when Alligator loses bullish alignment (jaws > teeth)
            if jaw_val > teeth_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when Alligator loses bearish alignment (teeth < lips)
            if teeth_val < lips_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish Alligator alignment: jaws < teeth < lips
            bullish_align = jaw_val < teeth_val and teeth_val < lips_val
            # Bearish Alligator alignment: jaws > teeth > lips
            bearish_align = jaw_val > teeth_val and teeth_val > lips_val
            
            # Long when: bullish alignment AND Bull Power > 0 AND price > 1d EMA50
            if bullish_align and bull_power_val > 0 and price > ema50_val:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: bearish alignment AND Bear Power < 0 AND price < 1d EMA50
            elif bearish_align and bear_power_val < 0 and price < ema50_val:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Alligator_ElderRay_1dEMA50"
timeframe = "6h"
leverage = 1.0