#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with Elder Ray (Bull/Bear Power) filter and volume spike confirmation
# Uses 1d timeframe for signals with 1w HTF for regime alignment to reduce false signals in choppy markets.
# Williams Alligator identifies trend absence (alligator sleeping) vs presence (alligator awakening).
# Elder Ray confirms trend strength: Bull Power > 0 and Bear Power < 0 for strong trends.
# Volume confirmation (2.0x 20-period average) ensures institutional participation.
# Designed for very low trade frequency (~30-100 total trades over 4 years) to minimize fee drag.
# Works in bull markets via trend-following signals, in bear via avoidance of weak/sideways market signals.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "1d_Alligator_ElderRay_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for regime filter (only trade in direction of weekly trend)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    # SMMA is similar to EMA but with different smoothing
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and Elder Ray)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping condition: all lines intertwined (market ranging)
        # Alligator awake: lines separated in proper order (trending)
        alligator_sleeping = (abs(jaw[i] - teeth[i]) < (close[i] * 0.001)) and \
                             (abs(teeth[i] - lips[i]) < (close[i] * 0.001)) and \
                             (abs(lips[i] - jaw[i]) < (close[i] * 0.001))
        
        # Alligator awake and trending up: Lips > Teeth > Jaw
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        
        # Alligator awake and trending down: Jaw > Teeth > Lips
        alligator_down = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator awake up + Bull Power > 0 + Bear Power < 0 + weekly uptrend + volume confirm
            if (alligator_up and bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake down + Bull Power < 0 + Bear Power > 0 + weekly downtrend + volume confirm
            elif (alligator_down and bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts sleeping or reverses down
            if alligator_sleeping or alligator_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping or reverses up
            if alligator_sleeping or alligator_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals