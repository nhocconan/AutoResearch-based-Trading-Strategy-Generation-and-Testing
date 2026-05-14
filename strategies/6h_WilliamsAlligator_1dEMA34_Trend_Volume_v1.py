#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via aligned SMAs.
# Long when LIPS > TEETH > JAW (bullish alignment), short when LIPS < TEETH < JAW (bearish).
# 1d EMA34 ensures higher-timeframe trend alignment to avoid counter-trend trades.
# Volume spike (2.0x 20-period average) confirms momentum and filters false signals.
# Discrete sizing 0.25 targets ~50-150 trades over 4 years (12-38/year) to minimize fee drag.
# Works in bull/bear via trend-following logic with strict alignment requirements.

name = "6h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Alligator alignment: bullish (L > T > J), bearish (L < T < J)
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculations)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(alligator_bullish[i]) or 
            np.isnan(alligator_bearish[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish Alligator + 1d uptrend + volume spike
            if alligator_bullish[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator + 1d downtrend + volume spike
            elif alligator_bearish[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish or trend reversal
            if alligator_bearish[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or trend reversal
            if alligator_bullish[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals