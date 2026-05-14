#!/usr/bin/env python3
"""
4h_ThreeBarReversal_WithVolume_TrendFilter
Hypothesis: Three-bar reversal patterns on 4h with volume confirmation and 1d EMA trend filter.
Buy when three consecutive higher closes form with volume spike and uptrend (price > EMA34).
Sell when three consecutive lower closes form with volume spike and downtrend (price < EMA34).
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing
reversal moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Three-bar reversal detection
    # Bullish: three consecutive higher closes
    bullish_reversal = (close > np.roll(close, 1)) & (np.roll(close, 1) > np.roll(close, 2))
    # Bearish: three consecutive lower closes
    bearish_reversal = (close < np.roll(close, 1)) & (np.roll(close, 1) < np.roll(close, 2))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(bullish_reversal[i]) or
            np.isnan(bearish_reversal[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        bull_rev = bullish_reversal[i]
        bear_rev = bearish_reversal[i]
        
        if position == 0:
            # Long: bullish reversal with volume spike and uptrend
            if bull_rev and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: bearish reversal with volume spike and downtrend
            elif bear_rev and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bearish reversal OR trend turns down
            if bear_rev or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bullish reversal OR trend turns up
            if bull_rev or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ThreeBarReversal_WithVolume_TrendFilter"
timeframe = "4h"
leverage = 1.0