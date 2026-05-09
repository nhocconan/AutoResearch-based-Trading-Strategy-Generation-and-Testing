#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator uses 3 SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# Long when: Lips > Teeth > Jaw (bullish alignment) + 1w EMA(34) rising + volume spike (>1.5x 20-period avg)
# Short when: Lips < Teeth < Jaw (bearish alignment) + 1w EMA(34) falling + volume spike
# Exit when: Alligator lines cross (trend reversal) OR price crosses the Teeth line
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 12-37 trades/year.
# Designed to work in both bull (trend following) and bear (mean-reversion at extremes) markets.

name = "12h_WilliamsAlligator_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    close_1w = df_1w['close']
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_prev = np.roll(ema_34_1w, 1)
    ema_34_1w_prev[0] = ema_34_1w[0]
    ema_rising = ema_34_1w > ema_34_1w_prev
    ema_falling = ema_34_1w < ema_34_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing
    # We'll use EMA as approximation for simplicity and performance
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment + 1w EMA rising + volume spike
            if (bullish_alignment[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + 1w EMA falling + volume spike
            elif (bearish_alignment[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment OR price crosses below Teeth
            if bearish_alignment[i] or (close[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment OR price crosses above Teeth
            if bullish_alignment[i] or (close[i] > teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals