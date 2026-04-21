#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) with price above/below all three lines as trend filter, combined with 4h Bollinger Band squeeze breakout for entry timing. In uptrend (price > Jaw > Teeth > Lips), buy when price breaks above upper BB; in downtrend (price < Jaw < Teeth < Lips), sell when price breaks below lower BB. Volume must exceed 1.5x 20-period average to confirm breakout. Exit on trend reversal or 2x ATR stop. Designed for 20-50 trades/year (80-200 total over 4 years) to minimize fee drag while capturing trend moves in both bull and bear markets. Williams Alligator identifies trending vs ranging markets, BB squeeze captures volatility breakouts, and volume confirmation reduces false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (3 SMAs)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 4h Bollinger Bands (20, 2)
    close = prices['close'].values
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Determine trend alignment: all three lines in order
        bullish_alignment = price_close > jaw_val and jaw_val > teeth_val and teeth_val > lips_val
        bearish_alignment = price_close < jaw_val and jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Enter long: bullish Alligator alignment + price breaks above upper BB + volume spike
            if bullish_alignment and price_close > bb_upper_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment + price breaks below lower BB + volume spike
            elif bearish_alignment and price_close < bb_lower_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (Alligator alignment breaks) OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit: Alligator alignment breaks
            if position == 1 and not bullish_alignment:
                exit_signal = True
            elif position == -1 and not bearish_alignment:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from approximate entry)
            if position == 1:
                entry_approx = bb_upper_aligned[i-1] if i > 0 else bb_upper[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                entry_approx = bb_lower_aligned[i-1] if i > 0 else bb_lower[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_BBBreakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0