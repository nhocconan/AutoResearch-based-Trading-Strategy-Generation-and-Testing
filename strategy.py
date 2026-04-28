#!/usr/bin/env python3
"""
1d_WilliamsAlligator_BullBear
Hypothesis: Williams Alligator (Jaw/Teeth/Lips SMAs) defines market regime: 
- Bull when Lips > Teeth > Jaw (green alignment) 
- Bear when Lips < Teeth < Jaw (red alignment)
Trades only in direction of Alligator alignment with volume confirmation and ATR volatility filter.
Avoids choppy markets by requiring clear alignment. Targets 10-25 trades/year on daily timeframe.
Works in bull (rides trends) and bear (shorts declines) by following Alligator's jaw alignment.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on weekly: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align weekly Alligator to daily
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Bullish (green) or Bearish (red)
        bullish_align = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_align = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volatility filter: avoid low volatility chop
        vol_filter = atr[i] > 0.5 * np.nanmedian(atr[max(0, i-50):i+1])
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.3 * vol_ma_20[i])
        
        # Entry: follow Alligator direction with filters
        long_entry = bullish_align and vol_filter and vol_confirm
        short_entry = bearish_align and vol_filter and vol_confirm
        
        # Exit: opposite Alligator alignment or loss of volume/volatility
        long_exit = bearish_align or (not vol_filter) or (not vol_confirm)
        short_exit = bullish_align or (not vol_filter) or (not vol_confirm)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_BullBear"
timeframe = "1d"
leverage = 1.0