#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator system with 12-hour Elder Ray for trend confirmation and volume spikes.
# Long when: Alligator jaws (13) < teeth (8) < lips (5) AND Elder Ray bull power > 0 AND volume spike.
# Short when: Alligator jaws > teeth > lips AND Elder Ray bear power < 0 AND volume spike.
# Uses Williams Alligator for trend identification, Elder Ray for bull/bear power confirmation, volume for momentum.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via Alligator alignment in uptrend and in bear markets via Alligator alignment in downtrend.
name = "4h_WilliamsAlligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (5, 8, 13 SMAs shifted)
    # Jaws: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars
    # Lips: 5-period SMA shifted 3 bars
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    jaws = np.roll(sma_13, 8)  # shifted 8 bars
    teeth = np.roll(sma_8, 5)  # shifted 5 bars
    lips = np.roll(sma_5, 3)   # shifted 3 bars
    
    # Alligator alignment: jaws < teeth < lips for uptrend, jaws > teeth > lips for downtrend
    alligator_up = (jaws < teeth) & (teeth < lips)
    alligator_down = (jaws > teeth) & (teeth > lips)
    
    # Load 12h data for Elder Ray (bull/bear power)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema_13_12h
    bear_power = low_12h - ema_13_12h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator up + Bull Power > 0 + volume spike
            long_condition = alligator_up[i] and (bull_power_aligned[i] > 0) and volume_spike[i]
            # Short: Alligator down + Bear Power < 0 + volume spike
            short_condition = alligator_down[i] and (bear_power_aligned[i] < 0) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator turns down or Bull Power <= 0
            if not alligator_up[i] or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator turns up or Bear Power >= 0
            if not alligator_down[i] or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals