#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Alligator + volume confirmation.
# Elder Ray measures bull/bear power vs EMA13. Alligator (JAW/TEETH/LIPS) defines trend regime.
# Long when Bull Power > 0, Bear Power < 0, price > Alligator TEETH, and volume > 1.5x MA.
# Short when Bear Power < 0, Bull Power > 0, price < Alligator TEETH, and volume > 1.5x MA.
# Uses discrete sizing 0.25 to limit drawdown. Target: 50-150 trades over 4 years.

name = "6h_ElderRay_Alligator_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Alligator components (SMAs with offsets)
    # JAW: 13-period SMA, offset 8 bars
    # TEETH: 8-period SMA, offset 5 bars
    # LIPS: 5-period SMA, offset 3 bars
    close_1d = df_1d['close'].values
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d EMA13 for Elder Ray (Bull/Bear Power)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h EMA13 for local trend alignment (optional filter)
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(13, 20) + 8  # 21 (for EMA13, volume MA, and Alligator offsets)
    
    for i in range(start_idx, n):
        if (np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or
            np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_ma_20[i]) or
            np.isnan(ema13_6h[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Elder Ray: Bull/Bear Power from 1d
        bull_power = bull_power_1d_aligned[i]   # High - EMA13
        bear_power = bear_power_1d_aligned[i]   # Low - EMA13
        
        # Alligator regime: price > TEETH = uptrend, price < TEETH = downtrend
        # Using 1d TEETH aligned to 6h
        uptrend_regime = curr_close > teeth_1d_aligned[i]
        downtrend_regime = curr_close < teeth_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power < 0 AND uptrend regime AND volume confirmation
            if bull_power > 0 and bear_power < 0 and uptrend_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND downtrend regime AND volume confirmation
            elif bear_power < 0 and bull_power > 0 and downtrend_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Bull Power <= 0 OR Bear Power >= 0 (loss of bullish momentum)
            if bull_power <= 0 or bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power >= 0 OR Bull Power <= 0 (loss of bearish momentum)
            if bear_power >= 0 or bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals