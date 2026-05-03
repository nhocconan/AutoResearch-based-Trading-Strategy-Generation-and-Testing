#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA(34) trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via SMAs with future shift
# Alligator sleeping (jaws/teeth/lips intertwined) = ranging market → avoid entries
# Alligator awakening (jaws > teeth > lips for long, reverse for short) = trending market
# 1d EMA(34) ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) confirms institutional participation
# Works in bull/bear markets by following Alligator direction in trending regimes
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 1d data (requires extra delay for confirmation)
    # Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMAs shifted forward
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMAs with minimum periods
    jaw_raw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth_raw = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips_raw = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Shift forward to avoid look-ahead (Alligator's predictive nature)
    jaw = np.roll(jaw_raw, -jaw_shift)
    teeth = np.roll(teeth_raw, -teeth_shift)
    lips = np.roll(lips_raw, -lips_shift)
    
    # Invalidate shifted values at the end
    jaw[-jaw_shift:] = np.nan
    teeth[-teeth_shift:] = np.nan
    lips[-lips_shift:] = np.nan
    
    # Align Alligator lines to 12h timeframe with extra delay for confirmation
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw, additional_delay_bars=1)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth, additional_delay_bars=1)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips, additional_delay_bars=1)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Alligator signals with 1d trend filter
        # Long: Jaw > Teeth > Lips (bullish alignment) + price above 1d EMA34 + volume spike
        # Short: Jaw < Teeth < Lips (bearish alignment) + price below 1d EMA34 + volume spike
        if position == 0:
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping (jaws/teeth/lips intertwined) OR price below 1d EMA34
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping OR price above 1d EMA34
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals