#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA50 trend filter and volume spike confirmation.
# Elder Ray measures bullish/bearish power: Bull Power = High - EMA, Bear Power = Low - EMA.
# Strong Bull Power + rising EMA indicates bullish momentum; strong Bear Power + falling EMA indicates bearish momentum.
# Combined with 12h EMA50 trend filter and volume spikes (>2x 20-period average), this captures institutional moves
# while avoiding chop. Designed for low trade frequency (~12-37/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend (12h EMA50).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Elder Ray calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 13-period EMA on 12h close for Elder Ray (EMA13)
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_12h - ema_13_12h  # Bull Power = High - EMA13
    bear_power = low_12h - ema_13_12h   # Bear Power = Low - EMA13
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe (waits for 12h bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Elder Ray conditions with trend filter
        # Strong bullish momentum: Bull Power > 0 AND rising + price above EMA50
        bullish_momentum = bull_power_val > 0 and bull_power_val > bull_power_aligned[i-1] and price > ema_val
        # Strong bearish momentum: Bear Power < 0 AND falling + price below EMA50
        bearish_momentum = bear_power_val < 0 and bear_power_val < bear_power_aligned[i-1] and price < ema_val
        
        if position == 0:
            # Long conditions: strong bullish momentum + volume spike
            if bullish_momentum and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: strong bearish momentum + volume spike
            elif bearish_momentum and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish momentum fades or price breaks below EMA50
                if not (bull_power_val > 0 and price > ema_val):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish momentum fades or price breaks above EMA50
                if not (bear_power_val < 0 and price < ema_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0