#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(close), Bear Power = Low - EMA(close).
# Enter long when Bull Power > 0 and rising (2-bar momentum) and price > 1d EMA34 (uptrend) and volume > 1.5x 20-bar average.
# Enter short when Bear Power < 0 and falling (2-bar momentum) and price < 1d EMA34 (downtrend) and volume > 1.5x 20-bar average.
# Exit when power reverses sign or volume drops below average.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Elder Ray measures bull/bear strength relative to trend; works in both regimes by confirming with higher-timeframe trend.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) for 6h Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(close)
    bear_power = low - ema_13   # Bear Power = Low - EMA(close)
    
    # Power momentum (2-bar change)
    bull_power_momentum = bull_power - np.roll(bull_power, 2)
    bear_power_momentum = bear_power - np.roll(bear_power, 2)
    # Handle first two bars
    bull_power_momentum[:2] = 0
    bear_power_momentum[:2] = 0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray entry conditions with momentum
        long_entry = (bull_power[i] > 0 and 
                     bull_power_momentum[i] > 0 and 
                     uptrend and 
                     vol_confirm)
                     
        short_entry = (bear_power[i] < 0 and 
                      bear_power_momentum[i] < 0 and 
                      downtrend and 
                      vol_confirm)
        
        # Exit conditions: power reverses or volume drops
        exit_long = (bull_power[i] <= 0) or (not vol_confirm)
        exit_short = (bear_power[i] >= 0) or (not vol_confirm)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals