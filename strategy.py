#/usr/bin/env python3
# 4h_1d_elder_ray_v1
# Strategy: 4h Elder Ray Power Index with volume confirmation and 1d EMA trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power via EMA. In bull markets, bull power > 0 and rising; in bear markets, bear power < 0 and falling. Combined with volume confirmation and 1d EMA trend filter, it captures strong trends while avoiding whipsaws in both bull and bear markets.
# Target: 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_elder_ray_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 13-period EMA for Elder Ray (standard setting)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High minus EMA
    bear_power = low - ema_13   # Low minus EMA
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: bull power > 0 and rising (bull power > previous bull power) + volume + above 1d EMA
        if (vol_confirmed and bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
            close[i] > ema_50_1d_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: bear power < 0 and falling (bear power < previous bear power) + volume + below 1d EMA
        elif (vol_confirmed and bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
              close[i] < ema_50_1d_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal or power weakening
        elif position == 1 and (bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals