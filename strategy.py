#!/usr/bin/env python3
"""
1d Cumulative Delta + Volume Spike + 1w Trend Filter
Hypothesis: Cumulative delta (buying minus selling pressure) shows institutional
accumulation/distribution. A spike in cumulative delta with volume confirms
strong directional moves. The 1-week EMA50 acts as a trend filter to avoid
counter-trend trades. Designed for low frequency (10-25 trades/year) to
minimize fee impact while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate cumulative delta: (close - low) - (high - close) = 2*close - high - low
    # This approximates buying pressure minus selling pressure per bar
    delta = 2 * close - high - low
    cum_delta = np.cumsum(delta)
    
    # Get weekly data for trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_w['close'].values
    ema_50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Volume spike detection (2x 20-period average to reduce frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Cumulative delta acceleration (change in delta) to detect spikes
    delta_change = np.diff(cum_delta, prepend=cum_delta[0])
    delta_ma = pd.Series(delta_change).rolling(window=20, min_periods=20).mean().values
    delta_std = pd.Series(delta_change).rolling(window=20, min_periods=20).std().values
    # Spike when delta change exceeds 2 standard deviations above mean
    delta_spike = delta_change > (delta_ma + 2.0 * delta_std)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_w_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(delta_ma[i]) or
            np.isnan(delta_std[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_w_aligned[i]
        
        if position == 0:
            # Long: cumulative delta spike with volume and above weekly EMA50
            if delta_spike[i] and volume_spike[i] and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: negative cumulative delta spike with volume and below weekly EMA50
            elif (-delta_spike[i] and volume_spike[i] and price < ema_50):
                # Negative delta spike: strong selling pressure
                negative_delta_spike = delta_change < (delta_ma - 2.0 * delta_std)
                if negative_delta_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: cumulative delta turns negative or price breaks below weekly EMA50
            if delta_change[i] < 0 or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: cumulative delta turns positive or price breaks above weekly EMA50
            if delta_change[i] > 0 or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_CumulativeDelta_VolumeSpike_1wEMA50"
timeframe = "1d"
leverage = 1.0