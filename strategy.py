#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (JAWS/TEETH/LIPS) with 1w trend filter and volume confirmation
# Alligator uses SMAs: JAWS(13,8), TEETH(8,5), LIPS(5,3) - measures trend strength via alignment
# In bull markets: JAWS > TEETH > LIPS (aligned up) + price above 1w EMA50 + volume spike → long
# In bear markets: JAWS < TEETH < LIPS (aligned down) + price below 1w EMA50 + volume spike → short
# Volume spike (>2.0x 20-period EMA) confirms institutional participation
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# Works in both regimes by measuring trend alignment strength

name = "6h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 6h data using SMAs
    # JAWS: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    # TEETH: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    # LIPS: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    jaws = np.roll(sma_13, 8)  # shift forward 8 bars
    teeth = np.roll(sma_8, 5)   # shift forward 5 bars
    lips = np.roll(sma_5, 3)    # shift forward 3 bars
    
    # Set NaN for invalid shifted values
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1w trend filter
        # Bullish alignment: JAWS > TEETH > LIPS (alligator mouth opening up)
        # Bearish alignment: JAWS < TEETH < LIPS (alligator mouth opening down)
        bullish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        bearish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        if position == 0:
            if bullish_alignment and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment breaks bearish OR price below 1w EMA50
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment breaks bullish OR price above 1w EMA50
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals