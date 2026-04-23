#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR volatility filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND close > 1d ATR(14) upper band AND volume > 1.5x average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND close < 1d ATR(14) lower band AND volume > 1.5x average.
Exit on Alligator alignment reversal or volatility contraction. Uses 12h timeframe targeting 50-150 total trades over 4 years.
Williams Alligator identifies trend phases, ATR bands filter for volatility expansion, volume confirms participation.
Designed to capture strong trending moves while avoiding whipsaws in low volatility or choppy markets across both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 12h: jaws=13, teeth=8, lips=5 (all SMMA)
    jaws, teeth, lips = compute_williams_alligator(high_12h, low_12h, close_12h)
    
    # Load 1d data for ATR volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR bands: upper = close + 1.5*ATR, lower = close - 1.5*ATR
    atr_upper_1d = close_1d + 1.5 * atr_1d
    atr_lower_1d = close_1d - 1.5 * atr_1d
    
    # Align HTF indicators to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    atr_upper_aligned = align_htf_to_ltf(prices, df_1d, atr_upper_1d)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1d, atr_lower_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_upper_aligned[i]) or 
            np.isnan(atr_lower_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaws_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr_upper_val = atr_upper_aligned[i]
        atr_lower_val = atr_lower_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Bullish Alligator alignment: jaws < teeth < lips
            bullish_align = jaws_val < teeth_val < lips_val
            # Bearish Alligator alignment: jaws > teeth > lips
            bearish_align = jaws_val > teeth_val > lips_val
            
            # Long: bullish alignment AND price > ATR upper band AND volume confirmation
            if bullish_align and price > atr_upper_val and vol_current > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: bearish alignment AND price < ATR lower band AND volume confirmation
            elif bearish_align and price < atr_lower_val and vol_current > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment turns bearish OR volatility contraction (price < ATR upper)
                bullish_align = jaws_val < teeth_val < lips_val
                if not bullish_align or price < atr_upper_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment turns bullish OR volatility contraction (price > ATR lower)
                bearish_align = jaws_val > teeth_val > lips_val
                if not bearish_align or price > atr_lower_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dATR_Volume"
timeframe = "12h"
leverage = 1.0