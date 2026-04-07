#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray system with 1w regime filter and volume confirmation
# Uses Elder Ray (Bull/Bear Power) to measure trend strength via EMA13
# - Bull Power = High - EMA13 (buying pressure)
# - Bear Power = Low - EMA13 (selling pressure)
# - 1w trend filter determines regime: only take Bull Power signals in uptrend, Bear Power in downtrend
# - Volume confirmation ensures institutional participation
# - Designed for low frequency (target: 15-30 trades/year) with clear entry/exit rules
# - Works in bull via Bull Power strength, in bear via Bear Power strength

name = "6h_elder_ray_1w_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA40)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 6-day EMA for Elder Ray (13-period on daily, approx 78 on 6h but we use 13 on 6h for responsiveness)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1w EMA
        uptrend = close[i] > ema_40_1w_aligned[i]
        downtrend = close[i] < ema_40_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when Bull Power turns negative or trend changes
            if bull_power[i] <= 0 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive or trend changes
            if bear_power[i] >= 0 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only take Bull Power signals in uptrend, Bear Power in downtrend
            if uptrend and vol_confirm:
                # Enter long when Bull Power is strong and rising
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif downtrend and vol_confirm:
                # Enter short when Bear Power is strong and falling (more negative)
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals