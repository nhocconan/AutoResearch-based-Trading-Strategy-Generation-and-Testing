#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12-hour Williams Alligator with 4-hour trend filter and volume confirmation
    # Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
    # In trending markets: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    # 4-hour EMA34 filters higher timeframe trend: only take longs in uptrend, shorts in downtrend
    # Volume spike confirms institutional participation
    # Targets ~20-30 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Williams Alligator SMAs on 12h
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values    # Red line (8)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values     # Green line (5)
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema34_4h_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend) + price above 4h EMA34 + volume spike
            if lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and \
               close[i] > ema34_4h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend) + price below 4h EMA34 + volume spike
            elif lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and \
                 close[i] < ema34_4h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines intertwine (no trend) or trend reverses vs 4h EMA34
            if position == 1:
                if not (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i]) or \
                   close[i] < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i]) or \
                   close[i] > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_4hEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0