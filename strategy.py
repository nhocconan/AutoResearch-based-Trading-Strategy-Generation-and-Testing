# 12h_1w_Camarilla_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: 12h price breaking above 1w Camarilla R1 or below S1 with volume confirmation and ATR-based stop.
# Uses 1w Camarilla levels as key support/resistance from higher timeframe structure.
# Volume filter (1.5x 20-period average) reduces false breakouts.
# ATR stop (2x ATR) manages risk during adverse moves.
# Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (reversals at key levels).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for Camarilla calculation (weekly high, low, close)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for 1w: R1, S1
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng = high_1w - low_1w
    camarilla_r1 = close_1w + rng * 1.1 / 12
    camarilla_s1 = close_1w - rng * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (using 12h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above 1w Camarilla R1 + volume surge
            if price > r1_12h[i] and price > r1_12h[i-1] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below 1w Camarilla S1 + volume surge
            elif price < s1_12h[i] and price < s1_12h[i-1] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 1w Camarilla S1 OR ATR stop hit (2*ATR)
            if price < s1_12h[i] or price < entry_price - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1w Camarilla R1 OR ATR stop hit (2*ATR)
            if price > r1_12h[i] or price > entry_price + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0