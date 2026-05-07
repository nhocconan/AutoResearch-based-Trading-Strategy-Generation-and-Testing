#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA21 trend
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    trend_up = close > ema_21_1d_aligned
    trend_down = close < ema_21_1d_aligned
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4*6h) to reduce trade frequency
    
    start_idx = max(1, 20)  # Ensure enough data for volume and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bull Power > 0 with volume spike in 1d uptrend
            if (bull_power[i] > 0 and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bear Power < 0 with volume spike in 1d downtrend
            elif (bear_power[i] < 0 and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Bull Power <= 0 or 1d trend changes to down
            if bull_power[i] <= 0 or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or 1d trend changes to up
            if bear_power[i] >= 0 or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull Power/Bear Power) measures the strength of bulls/bears relative to EMA13.
# Long when Bull Power > 0 (bulls in control) with volume spike and 1d uptrend.
# Short when Bear Power < 0 (bears in control) with volume spike and 1d downtrend.
# Works in bull markets (sustained Bull Power > 0) and bear markets (sustained Bear Power < 0).
# Volume spike confirms institutional participation. 6h timeframe reduces noise vs lower TFs.
# Discrete position sizing (0.25) balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.