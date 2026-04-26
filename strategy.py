#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation. In trending markets (price > 1d EMA50), long R1 breakout or short S1 breakout with volume > 1.5x 20-period average. Uses discrete position sizing (0.25) to minimize fee churn. Designed for 50-150 trades over 4 years by requiring confluence of breakout, trend, and volume. Works in bull/bear via trend filter: only takes breakouts in direction of 1d trend.
"""

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
    
    # Load 1d data ONCE before loop for HTF trend and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla pivot levels from previous 1d bar
        # Need previous 1d bar's high, low, close
        # Since we're on 12h timeframe, we need to get the completed 1d bar
        # that ended before current 12h bar
        # We'll use the aligned 1d data to get previous day's OHLC
        
        # Get index of 1d data for current time (already aligned)
        # We need to access df_1d values directly for pivot calculation
        # Find the 1d bar index that corresponds to current time
        # Since align_htf_to_ltf gives us values aligned to each 12h bar,
        # we need to reconstruct the 1d OHLC for pivot calculation
        
        # Simpler approach: calculate pivots on 1d data then align
        # Calculate typical price for 1d
        typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
        # Camarilla levels
        R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])
        S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])
        
        # Align these levels to 12h timeframe
        R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
        S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 with volume spike
            if close[i] > R1_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below S1 with volume spike
            if close[i] < S1_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0