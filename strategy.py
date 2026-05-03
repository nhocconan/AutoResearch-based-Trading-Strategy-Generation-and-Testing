#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA34 trend filter and volume confirmation
# Bull Power = High - EMA13 (1d), Bear Power = EMA13 (1d) - Low
# Long when Bull Power > 0 with volume > 1.5x 24-bar average and close > 1d EMA34 (uptrend)
# Short when Bear Power > 0 with volume > 1.5x 24-bar average and close < 1d EMA34 (downtrend)
# Exit when power crosses zero (momentum shift)
# Elder Ray measures bull/bear strength relative to EMA; works in trends with pullbacks and ranging markets.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "6h_ElderRay_Power_1dEMA34_Volume_v2"
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
    
    # Load 1d data ONCE before loop for EMA13 trend filter and power calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray Power
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray Power components
    bull_power = high - ema_13_aligned  # High - EMA13
    bear_power = ema_13_aligned - low   # EMA13 - Low
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 24) + 1  # EMA13(1d) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bulls in control) with volume spike and close > 1d EMA13 (uptrend)
            if (bull_power[i] > 0 and 
                volume_spike[i] and close[i] > ema_13_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 (bears in control) with volume spike and close < 1d EMA13 (downtrend)
            elif (bear_power[i] > 0 and 
                  volume_spike[i] and close[i] < ema_13_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power crosses below 0 (bulls losing control)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power crosses below 0 (bears losing control)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals