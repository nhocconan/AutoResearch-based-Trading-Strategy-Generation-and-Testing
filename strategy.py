#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# When Lips > Teeth > Jaw = bullish alignment; Lips < Teeth < Jaw = bearish.
# 1w EMA50 filter ensures we trade only in the direction of the weekly trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (bullish alignment with uptrend) and bear markets (bearish alignment with downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator SMAs
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # 13-period SMA (Jaw)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values   # 8-period SMA (Teeth)
    lips = close_s.rolling(window=5, min_periods=5).mean().values    # 5-period SMA (Lips)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Williams Alligator signals
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish alignment AND uptrend AND volume spike
            if bullish_alignment and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND downtrend AND volume spike
            elif bearish_alignment and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: alignment breaks or trend reverses
            if not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks or trend reverses
            if not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals