#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h Trend Filter and Volume Spike
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA)
# - Bullish when Bull Power > 0 and Bear Power < 0 (both conditions)
# - Bearish when Bear Power < 0 and Bull Power < 0 (both conditions)
# - Trend filter: 12h EMA34 slope (up/down) to avoid counter-trend trades
# - Volume spike: current volume > 2.0x 20-period average for confirmation
# - Works in bull/bear by using 12h trend filter to align with higher timeframe trend
# - Target: 15-30 trades/year on 6h to minimize fee drag (60-120 total over 4 years)

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray components on 6h timeframe
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (both true) + 12h uptrend + volume spike
            long_cond = (bull_power[i] > 0 and 
                        bear_power[i] < 0 and
                        ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Bear Power < 0 AND Bull Power < 0 (both true) + 12h downtrend + volume spike
            short_cond = (bear_power[i] < 0 and 
                         bull_power[i] < 0 and
                         ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power >= 0 (failure of bear power condition) or trend change
            if bear_power[i] >= 0 or ema_34_12h_aligned[i] <= ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power >= 0 (failure of bull power condition) or trend change
            if bull_power[i] >= 0 or ema_34_12h_aligned[i] >= ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals