# 1H_Camarilla_R1_S1_4hTrend_Volume
# Hypothesis: 1h strategy using daily Camarilla R1/S1 levels with 4h trend filter and volume confirmation.
# Enters long when price breaks above daily R1, close > 4h EMA34 (uptrend), and volume > 2x average.
# Enters short when price breaks below daily S1, close < 4h EMA34 (downtrend), and volume > 2x average.
# Exits when price returns to opposite S1/R1 level. Uses daily structure for direction, 1h for timing.
# Target: 15-30 trades/year per symbol to avoid fee drag. Works in bull/bear via 4h trend filter.

name = "1H_Camarilla_R1_S1_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and 4h for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) == 0 or len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous daily period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate R1 and S1 (inner levels)
    hl_range = high_1d - low_1d
    r1_1d = close_1d + 1.1 * hl_range / 12
    s1_1d = close_1d - 1.1 * hl_range / 12
    
    # Align levels to 1h timeframe (use previous daily period's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate EMA34 for trend filter (4h)
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R1, price above 4h EMA34 (uptrend), volume spike (>2x)
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below daily S1, price below 4h EMA34 (downtrend), volume spike (>2x)
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily S1 (opposite level)
            if close[i] <= s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to or above daily R1 (opposite level)
            if close[i] >= r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals