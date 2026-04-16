#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA trend filter and 1d volume spike.
# Long when Bull Power > 0 AND price > 12h EMA34 AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND price < 12h EMA34 AND volume > 1.5x 20-period average.
# Exit when power crosses zero (Bull Power <= 0 for long, Bear Power >= 0 for short).
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength relative to EMA,
# 12h EMA34 provides trend filter, volume confirmation reduces false signals.
# Target: 80-160 total trades over 4 years (20-40/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA(34) for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data once before loop for Elder Ray calculation (using 13-period EMA as standard)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Elder Ray Index (Bull Power/Bear Power) ===
    # Standard Elder Ray uses 13-period EMA
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        ema_val = ema_34_12h_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Calculate 20-period volume average
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_20 = 0.0
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power <= 0 (bulls losing strength)
            if bull_val <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power >= 0 (bears losing strength)
            if bear_val >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND price > 12h EMA34 AND volume confirmation
            if bull_val > 0 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND price < 12h EMA34 AND volume confirmation
            elif bear_val < 0 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_1dVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0