#!/usr/bin/env python3
name = "6h_Structure_Swing_Retest"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily swing points: 5-bar fractals (need 2 bars confirmation each side)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate swing highs and lows
    swing_high = np.full(len(high_1d), np.nan)
    swing_low = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = low_1d[i]
    
    # Forward fill to get last swing level
    swing_high_series = pd.Series(swing_high)
    swing_low_series = pd.Series(swing_low)
    last_swing_high = swing_high_series.ffill().bfill().values
    last_swing_low = swing_low_series.ffill().bfill().values
    
    # Align swing levels to 6h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, last_swing_high, additional_delay_bars=2)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, last_swing_low, additional_delay_bars=2)
    
    # Daily trend: EMA(50) on close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: retest of swing low in uptrend with volume
            if (close[i] > swing_low_aligned[i] * 1.001 and  # slight penetration
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # daily uptrend
                volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: retest of swing high in downtrend with volume
            elif (close[i] < swing_high_aligned[i] * 0.999 and  # slight penetration
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # daily downtrend
                  volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend reversal or failure to hold above swing low
            if (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] or  # trend change
                close[i] < swing_low_aligned[i] * 0.995):  # break below swing low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or failure to hold below swing high
            if (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] or  # trend change
                close[i] > swing_high_aligned[i] * 1.005):  # break above swing high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s swing retest strategy
# - Uses daily swing points (5-bar fractals) as key support/resistance levels
# - Long when price retests and holds above daily swing low in uptrend with volume
# - Short when price retests and holds below daily swing high in downtrend with volume
# - Swing points are structural levels that work in both trending and ranging markets
# - Requires volume confirmation (1.5x average) to avoid false breakouts
# - Exits on trend reversal or break of the swing level
# - Targets 15-35 trades/year by requiring confluence of trend, level, and volume
# - Works in bull markets (buying swing low retests) and bear markets (selling swing high retests)