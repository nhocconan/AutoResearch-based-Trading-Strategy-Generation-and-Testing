#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Williams Alligator trend filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Williams Alligator (Jaw/Teeth/Lips SMAs) defines trend: aligned = trending, entwined = ranging
# Enter long when Bull Power > 0 AND Alligator aligned (Lips > Teeth > Jaw) AND volume > 1.5x 20 EMA
# Enter short when Bear Power < 0 AND Alligator aligned (Lips < Teeth < Jaw) AND volume spike
# Exit when power reverses or Alligator disentangles
# Designed for low frequency: ~15-30 trades/year with 0.25 sizing
# Works in bull/bear by following Alligator trend; Elder Ray provides precise entry timing
# Volume confirmation filters weak breakouts

name = "6h_ElderRay_Alligator_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values   # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 6h EMA13 for Elder Ray power
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need EMA13 (13) + Alligator Lips (5) + volume EMA20 (20)
    start_idx = max(13, 5, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment conditions
        # Aligned up (trending up): Lips > Teeth > Jaw
        aligned_up = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Aligned down (trending down): Lips < Teeth < Jaw
        aligned_down = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            if aligned_up and bull_power[i] > 0 and volume_spike[i]:
                # Long: Alligator aligned up + bullish power + volume
                signals[i] = 0.25
                position = 1
            elif aligned_down and bear_power[i] < 0 and volume_spike[i]:
                # Short: Alligator aligned down + bearish power + volume
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0  # No clear signal
        
        elif position == 1:  # Long position
            # Exit: Power turns bearish OR Alligator disentangles
            if bull_power[i] <= 0 or not aligned_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Power turns bullish OR Alligator disentangles
            if bear_power[i] >= 0 or not aligned_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals