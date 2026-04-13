#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter + volume confirmation
    # Long: price > Donchian(20) high AND ATR(14) > ATR(50) AND volume > 1.5x 20-period average
    # Short: price < Donchian(20) low AND ATR(14) > ATR(50) AND volume > 1.5x 20-period average
    # Exit: opposite Donchian breakout
    # Using 4h timeframe for optimal trade frequency (target 20-50/year), ATR regime to avoid low-vol whipsaws,
    # and volume spike confirmation to avoid false breakouts. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = np.zeros(n)
    tr1[0] = high[0] - low[0]
    for i in range(1, n):
        tr1[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr14 = np.full(n, np.nan)
    atr50 = np.full(n, np.nan)
    
    # ATR(14)
    for i in range(14, n):
        if i == 14:
            atr14[i] = np.mean(tr1[1:15])
        else:
            atr14[i] = (atr14[i-1] * 13 + tr1[i]) / 14
    
    # ATR(50)
    for i in range(50, n):
        if i == 50:
            atr50[i] = np.mean(tr1[1:51])
        else:
            atr50[i] = (atr50[i-1] * 49 + tr1[i]) / 50
    
    atr_regime = atr14 > atr50  # High volatility regime
    
    # Get 4h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_regime[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry logic: Breakout + ATR regime + volume confirmation
        long_entry = long_breakout and atr_regime[i] and volume_spike[i]
        short_entry = short_breakout and atr_regime[i] and volume_spike[i]
        
        # Exit logic: opposite breakout
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_breakout_atr_regime_volume_v1"
timeframe = "4h"
leverage = 1.0