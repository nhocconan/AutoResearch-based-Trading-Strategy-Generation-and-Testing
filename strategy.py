#!/usr/bin/env python3

# 6h_ElderRay_ZoneRecovery_v1
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with zone recovery logic. 
# In strong trends (ADX>25), price often pulls back to EMA13 before resuming.
# Long when Bear Power crosses above zero (bulls taking control) in uptrend.
# Short when Bull Power crosses below zero (bears taking control) in downtrend.
# Uses 1d EMA50 for trend filter and 1d volume spike for confirmation.
# Designed to capture trend resumption moves with low frequency and high win rate.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_ZoneRecovery_v1"
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
    
    # Get 1d data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6-day EMA for Elder Ray (13-period EMA on 6h closes)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 / (13 + 1)) + (ema13[i-1] * (11 / (13 + 1)))
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema50_1d[i-1] * (49 / (50 + 1)))
    
    # 1d volume spike: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma20_1d[i] = np.mean(vol_1d[i-20:i])
    vol_spike_1d = vol_1d > (1.5 * vol_ma20_1d)
    
    # Align 1d indicators to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # Prevent overtrading (approx 1 day)
    
    start_idx = max(20, 50)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend: price vs EMA50
        trend_1d_up = close[i] > ema50_1d_aligned[i]
        trend_1d_down = close[i] < ema50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bear Power crosses above zero (bulls taking control) in uptrend with volume spike
            if (bear_power[i] > 0 and bear_power[i-1] <= 0 and 
                trend_1d_up and 
                vol_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bull Power crosses below zero (bears taking control) in downtrend with volume spike
            elif (bull_power[i] < 0 and bull_power[i-1] >= 0 and 
                  trend_1d_down and 
                  vol_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: Bull Power crosses below zero (loss of bullish momentum)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero (loss of bearish momentum)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals