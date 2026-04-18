#!/usr/bin/env python3
"""
4h Elder Ray Index with 1w Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) identifies institutional buying/selling pressure.
Combined with 1-week EMA trend filter and volume confirmation, it captures strong momentum moves while avoiding whipsaws.
Works in bull markets via Bull Power expansion and in bear markets via Bear Power expansion. Low trade frequency due to
requirement for both power expansion and volume confirmation.
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
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate EMA13 for Elder Ray (using close prices)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(ema13[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1w_aligned[i]
        vol_ok = vol_confirm[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Enter long: Bull Power expanding (increasing) with volume + uptrend
            if i > 0 and not np.isnan(bull_power[i-1]):
                bull_expanding = bull > bull_power[i-1]
                if bull_expanding and vol_ok and bull > 0 and close[i] > trend:
                    signals[i] = 0.25
                    position = 1
            # Enter short: Bear Power expanding (decreasing, i.e., becoming more negative) with volume + downtrend
            elif i > 0 and not np.isnan(bear_power[i-1]):
                bear_expanding = bear < bear_power[i-1]  # More negative = stronger selling
                if bear_expanding and vol_ok and bear < 0 and close[i] < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Bull Power contracting or trend turns down
            if i > 0 and not np.isnan(bull_power[i-1]):
                bull_contracting = bull < bull_power[i-1]
                if bull_contracting or close[i] < trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power contracting or trend turns up
            if i > 0 and not np.isnan(bear_power[i-1]):
                bear_contracting = bear > bear_power[i-1]  # Less negative = weaker selling
                if bear_contracting or close[i] > trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Elder_Ray_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0