#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend absence when lines intertwine
# Trade when price breaks above/below Alligator with alignment to 12h EMA50 trend and volume spike
# Works in ranging markets by capturing breakouts from consolidation and in trends by filtering counter-trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60  # Need enough data for Alligator lines
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator lines intertwined (no trend) when jaws < teeth < lips or reverse
        alligator_entwined = (jaw[i] < teeth[i] < lips[i]) or (jaw[i] > teeth[i] > lips[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Alligator with bullish 12h trend and volume spike
            if (close[i] > lips[i] and close[i-1] <= lips[i-1] and  # Just broke above lips
                close[i] > ema_50_12h_aligned[i] and               # Above 12h EMA50 (bullish trend)
                volume_spike[i] and                              # Volume confirmation
                alligator_entwined):                             # Only trade after consolidation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Alligator with bearish 12h trend and volume spike
            elif (close[i] < jaw[i] and close[i-1] >= jaw[i-1] and  # Just broke below jaw
                  close[i] < ema_50_12h_aligned[i] and              # Below 12h EMA50 (bearish trend)
                  volume_spike[i] and                               # Volume confirmation
                  alligator_entwined):                              # Only trade after consolidation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price re-enters Alligator (teeth) OR 12h trend turns bearish
            if close[i] < teeth[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price re-enters Alligator (teeth) OR 12h trend turns bullish
            if close[i] > teeth[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals