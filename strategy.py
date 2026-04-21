#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot point reversal with 1d VWAP trend filter and volume spike.
In ranging markets, price tends to revert to mean after touching Camarilla support/resistance levels.
During trends, price breaks through these levels with continuation. Uses 1d VWAP to filter trend
direction and avoid counter-trend trades. Volume spike confirms institutional interest at key levels.
Designed for ~25-40 trades/year to minimize fee drag while capturing reversals in ranging markets
and breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_prev = high_prev - low_prev
    camarilla_h5 = close_prev + range_prev * 1.1 / 2
    camarilla_h4 = close_prev + range_prev * 1.1 / 4
    camarilla_h3 = close_prev + range_prev * 1.1 / 6
    camarilla_l3 = close_prev - range_prev * 1.1 / 6
    camarilla_l4 = close_prev - range_prev * 1.1 / 4
    camarilla_l5 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # 1d VWAP for trend filter (volume-weighted average price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vwap_val = vwap_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        vol_threshold = 1.8  # Volume must be 1.8x average for confirmation
        
        if position == 0:
            # Long reversal: price touches/slightly breaks L3 support with volume, above VWAP (uptrend bias)
            if (price_close <= camarilla_l3_aligned[i] * 1.002 and  # Allow slight penetration
                vol_ratio_val > vol_threshold and 
                price_close > vwap_val):
                signals[i] = 0.25
                position = 1
            # Short reversal: price touches/slightly breaks H3 resistance with volume, below VWAP (downtrend bias)
            elif (price_close >= camarilla_h3_aligned[i] * 0.998 and  # Allow slight penetration
                  vol_ratio_val > vol_threshold and 
                  price_close < vwap_val):
                signals[i] = -0.25
                position = -1
            # Long breakout: price breaks above H4 with strong volume, above VWAP
            elif (price_close > camarilla_h4_aligned[i] and 
                  vol_ratio_val > vol_threshold * 1.5 and  # Higher threshold for breakout
                  price_close > vwap_val):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below L4 with strong volume, below VWAP
            elif (price_close < camarilla_l4_aligned[i] and 
                  vol_ratio_val > vol_threshold * 1.5 and  # Higher threshold for breakdown
                  price_close < vwap_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches VWAP (mean reversion) or breaks below L3 (failed bounce)
                if (price_close >= vwap_val * 0.998 or  # Reached VWAP
                    price_close < camarilla_l3_aligned[i] * 0.995):  # Broke below support
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            else:  # position == -1
                # Exit short: price reaches VWAP (mean reversion) or breaks above H3 (failed rejection)
                if (price_close <= vwap_val * 1.002 or  # Reached VWAP
                    price_close > camarilla_h3_aligned[i] * 1.005):  # Broke above resistance
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "4h_Camarilla_VWAP_Reversal_Breakout_Volume"
timeframe = "4h"
leverage = 1.0