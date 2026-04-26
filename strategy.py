#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d EMA50 regime filter. 
Long when Bull Power > 0 AND 1d close > EMA50 (bull regime). 
Short when Bear Power > 0 AND 1d close < EMA50 (bear regime). 
Volume confirmation (>1.5x 20-bar MA) filters weak breakouts. 
Elder Ray measures underlying buying/selling pressure beyond price, effective in both bull/bear markets via regime alignment.
Target: 15-30 trades/year (60-120 total over 4 years).
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
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's close for EMA50 regime filter
    close_1d_vals = df_1d['close'].values
    
    # 1d EMA50 for regime filter (bull if close > EMA50, bear if close < EMA50)
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # EMA13 for Elder Ray calculation (on 6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Index components
    bull_power = high - ema_13      # Buying strength: ability to push price above EMA13
    bear_power = ema_13 - low       # Selling strength: ability to push price below EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for 1d EMA, 13 for EMA13, 20 for volume)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime conditions
        bull_regime = close_val > ema_50_val   # 1d regime: bullish
        bear_regime = close_val < ema_50_val   # 1d regime: bearish
        
        # Entry conditions: Elder Ray alignment with regime + volume confirmation
        long_entry = (bull_val > 0) and bull_regime and vol_spike
        short_entry = (bear_val > 0) and bear_regime and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when bull power fades OR regime changes
            if (bull_val <= 0) or (not bull_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short - exit when bear power fades OR regime changes
            if (bear_val <= 0) or (not bear_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime"
timeframe = "6h"
leverage = 1.0