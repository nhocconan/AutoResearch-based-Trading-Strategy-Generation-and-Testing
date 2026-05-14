#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong bullish momentum when Bull Power > 0 AND rising, strong bearish when Bear Power < 0 AND falling
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume spike (>1.5x 20-period average) confirms significant participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure
# Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe

name = "6h_ElderRay_12hEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # 12h EMA50, EMA13, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        vol_spike = curr_volume > 1.5 * curr_vol_ma
        
        # Elder Ray momentum conditions
        bull_momentum = curr_bull_power > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        bear_momentum = curr_bear_power < 0 and (i == start_idx or bear_power[i] < bear_power[i-1])
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: bear momentum OR price crosses below 12h EMA50
            if bear_momentum or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull momentum OR price crosses above 12h EMA50
            if bull_momentum or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish momentum AND above 12h EMA50 AND volume spike
            if bull_momentum and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish momentum AND below 12h EMA50 AND volume spike
            elif bear_momentum and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals