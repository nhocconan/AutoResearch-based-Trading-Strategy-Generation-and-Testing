#!/usr/bin/env python3
"""
6h_ElderRay_Alligator_Combo
Hypothesis: Combine Elder Ray (Bull/Bear Power) with Williams Alligator on 6h timeframe using 1d/1w HTF filters.
Elder Ray measures bull/bear power via EMA(13). Alligator uses SMAs(5,8,13) to identify trend direction and strength.
Long when: Bull Power > 0, Bear Power < 0, price > Alligator Jaw (Teeth > Lips aligned), 1d trend up.
Short when: Bear Power < 0, Bull Power < 0, price < Alligator Jaw (Teeth < Lips aligned), 1d trend down.
Volume confirmation (>1.5x average) reduces false signals. Discrete sizing 0.25 targets ~20-30 trades/year.
Designed for both bull and bear markets: Alligator identifies trend, Elder Ray measures power, volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Williams Alligator: Jaw=SMA(13), Teeth=SMA(8), Lips=SMA(5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA13(13), Alligator jaws(13), volume MA(20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        # Alligator alignment: Teeth > Lips > Jaw (bullish) or Teeth < Lips < Jaw (bearish)
        bullish_alignment = teeth_val > lips_val and lips_val > jaw_val
        bearish_alignment = teeth_val < lips_val and lips_val < jaw_val
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, bullish alignment, 1d uptrend, volume confirmation
            long_signal = (bull_val > 0) and (bear_val < 0) and bullish_alignment and (close_val > ema_50_1d_val) and volume_confirmed
            # Short: Bear Power < 0, Bull Power < 0, bearish alignment, 1d downtrend, volume confirmation
            short_signal = (bear_val < 0) and (bull_val < 0) and bearish_alignment and (close_val < ema_50_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power becomes positive (bulls weakening) OR price closes below Alligator Jaw
            if bear_val > 0 or close_val < jaw_val:
                signals[i] = 0.0
                position = 0
            # Exit: 1d trend reversal
            elif close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power becomes negative (bears weakening) OR price closes above Alligator Jaw
            if bull_val < 0 or close_val > jaw_val:
                signals[i] = 0.0
                position = 0
            # Exit: 1d trend reversal
            elif close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Alligator_Combo"
timeframe = "6h"
leverage = 1.0