# 1D_CAMARILLA_R1_S1_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION
# Hypothesis: On 1d timeframe, use weekly (1w) Camarilla R1/S1 levels as breakout triggers.
# Enter long when price breaks above R1 with volume spike and weekly uptrend (close > EMA34 weekly).
# Enter short when price breaks below S1 with volume spike and weekly downtrend (close < EMA34 weekly).
# Exit when price returns to the opposite level (S1 for longs, R1 for shorts).
# Targets 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
# Uses 1w trend filter and volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets via trend filter.

name = "1D_CAMARILLA_R1_S1_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Camarilla levels from weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R1, S1
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas: R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    R1 = close_1w + (high_1w - low_1w) * 1.1 / 2
    S1 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # 1w EMA for trend filter (34-period)
    ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and weekly uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and weekly downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S1 level
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R1 level
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals