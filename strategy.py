#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 6h
# - 1d EMA(50) trend filter: only take longs in uptrend (price > EMA50), shorts in downtrend (price < EMA50)
# - Volume spike confirmation: current volume > 2.0 * 20-period volume average reduces false signals
# - Discrete position sizing: 0.25 for entries, 0.0 for exits
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid fee drag
# - Mean reversion works well in ranging markets (2022-2024), trend filter helps capture momentum in bull/bear regimes

name = "6h_1d_williamsr_meanrev_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h Williams %R(14) for mean reversion signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R returns above -20 (overbought) or trend change
            if williams_r[i] > -20 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R returns below -80 (oversold) or trend change
            if williams_r[i] < -80 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume confirmation and trend filter
            if williams_r[i] < -80 and volume_confirm[i] and close[i] > ema_50_aligned[i]:
                # Oversold + volume spike + uptrend = long
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and volume_confirm[i] and close[i] < ema_50_aligned[i]:
                # Overbought + volume spike + downtrend = short
                position = -1
                signals[i] = -0.25
    
    return signals