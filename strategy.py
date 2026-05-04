#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 1d close > 1d EMA34 (uptrend) AND volume > 2.0x 20 EMA
# Short when jaws < teeth < lips AND 1d close < 1d EMA34 (downtrend) AND volume > 2.0x 20 EMA
# Uses 12h for Alligator calculation (Williams Alligator is a trend-following indicator using smoothed moving averages),
# 1d for trend direction to avoid counter-trend trades, and volume spike for confirmation.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-25 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Williams Alligator is effective in trending markets and avoids whipsaws in ranging markets.

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
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
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    uptrent_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate 12h Williams Alligator (using SMMA - Smoothed Moving Average)
    # SMMA is similar to EMA but with different smoothing
    # We'll use EMA as approximation for SMMA since true SMMA requires recursive calculation
    # Jaws: 13-period SMMA (approximated with EMA)
    # Teeth: 8-period SMMA (approximated with EMA)
    # Lips: 5-period SMMA (approximated with EMA)
    jaws = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator conditions: jaws > teeth > lips for uptrend, jaws < teeth < lips for downtrend
    alligator_long = (jaws > teeth) & (teeth > lips)
    alligator_short = (jaws < teeth) & (teeth < lips)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for Alligator calculation
        # Skip if any value is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(uptrent_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator long alignment AND 1d uptrend AND volume spike
            if (alligator_long[i] and 
                uptrent_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator short alignment AND 1d downtrend AND volume spike
            elif (alligator_short[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator short alignment OR 1d trend changes to downtrend
            if (alligator_short[i] or 
                downtrent_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator long alignment OR 1d trend changes to uptrend
            if (alligator_long[i] or 
                uptrent_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals