#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 6h Williams %R for mean reversion entries
# In ranging markets (common in 2025+ BTC/ETH), price reverts to the mean within the prevailing trend.
# Williams %R identifies overbought/oversold conditions on 6h chart.
# 12h Supertrend ensures we only take mean-reversion trades in the direction of the higher-timeframe trend.
# Designed for low trade frequency (<25/year) to minimize fee drag in both bull and bear markets.

name = "6h_WilliamsR_MeanReversion_12hSupertrend_v1"
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
    
    # Load 12h data ONCE before loop for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_basic_12h = hl2_12h + 3.0 * atr_12h
    lower_basic_12h = hl2_12h - 3.0 * atr_12h
    
    # Final Upper and Lower Bands
    upper_final_12h = np.zeros_like(close_12h)
    lower_final_12h = np.zeros_like(close_12h)
    upper_final_12h[0] = upper_basic_12h[0]
    lower_final_12h[0] = lower_basic_12h[0]
    
    for i in range(1, len(close_12h)):
        if close_12h[i-1] <= upper_final_12h[i-1]:
            upper_final_12h[i] = min(upper_basic_12h[i], upper_final_12h[i-1])
        else:
            upper_final_12h[i] = upper_basic_12h[i]
            
        if close_12h[i-1] >= lower_final_12h[i-1]:
            lower_final_12h[i] = max(lower_basic_12h[i], lower_final_12h[i-1])
        else:
            lower_final_12h[i] = lower_basic_12h[i]
    
    # Supertrend direction: 1 = uptrend, -1 = downtrend
    supertrend_12h = np.ones_like(close_12h)
    supertrend_12h[0] = 1
    for i in range(1, len(close_12h)):
        if close_12h[i] <= upper_final_12h[i]:
            supertrend_12h[i] = 1
        elif close_12h[i] >= lower_final_12h[i]:
            supertrend_12h[i] = -1
        else:
            supertrend_12h[i] = supertrend_12h[i-1]
            if supertrend_12h[i] == 1 and close_12h[i] > upper_final_12h[i]:
                supertrend_12h[i] = -1
            elif supertrend_12h[i] == -1 and close_12h[i] < lower_final_12h[i]:
                supertrend_12h[i] = 1
    
    # Align 12h Supertrend to 6h timeframe
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    
    # Calculate ATR(10) for dynamic stoploss on 6h
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.max([tr1_6h[0], tr2_6h[0], tr3_6h[0]])], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6h = pd.Series(tr_6h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 14  # warmup for Williams %R
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_williams_r = williams_r_6h[i]
        curr_atr = atr_6h[i]
        curr_supertrend = supertrend_12h_aligned[i]
        
        if position == 0:  # Flat - look for mean reversion entries
            # Only trade in direction of 12h Supertrend trend
            if curr_supertrend == 1:  # 12h uptrend - look for oversold bounces
                if curr_williams_r < -80:  # Oversold condition
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            elif curr_supertrend == -1:  # 12h downtrend - look for overbought bounces
                if curr_williams_r > -20:  # Overbought condition
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R returns to neutral territory (-50)
            elif curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: Williams %R returns to neutral territory (-50)
            elif curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals