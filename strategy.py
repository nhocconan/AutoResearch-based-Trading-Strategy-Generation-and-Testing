#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets: buy when Bull Power > 0 and rising + price > 1d EMA50 + volume spike
# In bear markets: sell when Bear Power < 0 and falling + price < 1d EMA50 + volume spike
# Works in both regimes by adapting to prevailing trend via 1d EMA50 filter
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # 1d EMA50, EMA13, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_ema_13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price closes below 1d EMA50
            if curr_bull_power <= 0 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR price closes above 1d EMA50
            if curr_bear_power >= 0 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND rising + above 1d EMA50 + volume confirmation
            bull_rising = curr_bull_power > bull_power[i-1] if i > 0 else False
            if (curr_bull_power > 0 and bull_rising and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND falling + below 1d EMA50 + volume confirmation
            elif (curr_bear_power < 0 and 
                  curr_bear_power < bear_power[i-1] if i > 0 else False and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals