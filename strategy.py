#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1w EMA50 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close)
# Trend filter: 1w EMA50 (primary trend direction)
# Entry conditions:
#   Long: Bull Power > 0 AND Bear Power < 0 AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
#   Short: Bull Power < 0 AND Bear Power > 0 AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit: Elder Ray divergence (Bull Power < 0 for long OR Bear Power > 0 for short) OR price crosses 1w EMA50
# Williams Alligator identifies trend presence and direction with less whipsaw than single MAs
# 1w EMA50 provides stronger trend filter than shorter HTF, reducing false signals in chop
# Volume spike confirms breakout strength and institutional participation
# Discrete position sizing: 0.25 for long/short to minimize fee churn while maintaining adequate exposure
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_ElderRay_1wEMA50_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA13 for Elder Ray (using close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Elder Ray divergence (Bull Power < 0) OR price below 1w EMA50
            if curr_bull < 0 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Elder Ray divergence (Bear Power > 0) OR price above 1w EMA50
            if curr_bear > 0 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND price > 1w EMA50 AND volume spike
            if (curr_bull > 0 and curr_bear < 0 and 
                curr_close > curr_ema_1w and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Bull Power < 0 AND Bear Power > 0 AND price < 1w EMA50 AND volume spike
            elif (curr_bull < 0 and curr_bear > 0 and 
                  curr_close < curr_ema_1w and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals