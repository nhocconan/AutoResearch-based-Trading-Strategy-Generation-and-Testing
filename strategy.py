#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + Daily RSI filter + Volume Spike
# Elder Ray measures bull/bear power relative to EMA. Bull Power = High - EMA, Bear Power = Low - EMA.
# We go long when Bull Power > 0 and Bear Power rising (less negative), short when Bear Power < 0 and Bull Power falling.
# Filtered by daily RSI (avoid extremes) and volume spikes to avoid chop.
# Works in bull/bear via adaptive EMA. Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) on daily close
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align EMA and RSI to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Elder Ray components using aligned EMA
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Smooth Elder Ray with EMA(5) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate change in smoothed power for momentum
    bull_power_change = np.diff(bull_power_smooth, prepend=bull_power_smooth[0])
    bear_power_change = np.diff(bear_power_smooth, prepend=bear_power_smooth[0])
    
    # Volume spike detection: volume > 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    vol_spike = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long conditions:
        # 1. Bull power positive (above EMA)
        # 2. Bear power increasing (becoming less negative)
        # 3. RSI not overbought (< 70) to avoid exhaustion
        # 4. Volume spike for confirmation
        if (bull_power_smooth[i] > 0 and
            bear_power_change[i] > 0 and
            rsi_aligned[i] < 70 and
            vol_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short conditions:
        # 1. Bear power negative (below EMA)
        # 2. Bull power decreasing (weakening)
        # 3. RSI not oversold (> 30) to avoid exhaustion
        # 4. Volume spike for confirmation
        elif (bear_power_smooth[i] < 0 and
              bull_power_change[i] < 0 and
              rsi_aligned[i] > 30 and
              vol_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: power divergence or RSI extreme
        elif position == 1 and (bull_power_smooth[i] < 0 or rsi_aligned[i] > 80):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_smooth[i] > 0 or rsi_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_RSI_VolumeSpike"
timeframe = "6h"
leverage = 1.0