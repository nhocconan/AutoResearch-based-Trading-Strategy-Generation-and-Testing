#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for trend filter and 6h Elder Ray (EMA13-based bull/bear power) for momentum
# Entry: Long when bull power > 0 AND price > 12h EMA50 (uptrend) AND volume spike
#        Short when bear power < 0 AND price < 12h EMA50 (downtrend) AND volume spike
# Exit: Close crosses 12h EMA50 (trend reversal) OR Elder Ray power crosses zero (momentum shift)
# Works in both bull and bear markets by trading with 12h trend using Elder Ray momentum
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_ElderRay_BullBearPower_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Elder Ray Bull/Bear Power (EMA13-based)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull power > 0 AND price > 12h EMA50 (uptrend) AND volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear power < 0 AND price < 12h EMA50 (downtrend) AND volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 12h EMA50 (trend change) OR bear power >= 0 (momentum loss)
            if close[i] < ema_50_12h_aligned[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 12h EMA50 (trend change) OR bull power <= 0 (momentum loss)
            if close[i] > ema_50_12h_aligned[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals