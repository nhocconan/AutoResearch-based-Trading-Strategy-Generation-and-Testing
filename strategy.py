#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Uses 6h timeframe for signal generation with Elder Ray calculated from 6h data
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via strong bull power + uptrend, in bear via strong bear power + downtrend
# Elder Ray captures momentum exhaustion better than RSI/WilliamsR in ranging markets

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Measures bull strength
    bear_power = low - ema_13   # Measures bear strength (negative values)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA13)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Strong bull power (>0) + price > 1d EMA34 + volume confirm
            if bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power (<0) + price < 1d EMA34 + volume confirm
            elif bear_power[i] < 0 and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull power turns negative (<0) or bear power becomes strong (< -1.0 * ATR proxy)
            # Use 0 as simple exit for bull power turning negative
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear power turns positive (>0) or bull power becomes strong (> 1.0 * ATR proxy)
            # Use 0 as simple exit for bear power turning positive
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals