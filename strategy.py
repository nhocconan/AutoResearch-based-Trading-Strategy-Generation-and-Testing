# 12h_1d_Camarilla_Pivot_Bounce_v2
# Hypothesis: Use daily Camarilla pivot levels with mean-reversion bounces on 12h.
# Long when price touches S3/S4 with bullish engulfing candle, short when touches R3/R4 with bearish engulfing candle.
# Camarilla levels are widely watched intraday levels that often act as support/resistance.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
# Works in range-bound markets via mean reversion at extremes, and in trending markets via breakout filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Bounce_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily Camarilla pivot levels
    # Camarilla formula: 
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # PP = (High + Low + Close)/3
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    camarilla_r4 = prev_close + range_val * 1.1 / 2
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    camarilla_s4 = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_array = np.full(len(df_1d), camarilla_r4)
    camarilla_r3_array = np.full(len(df_1d), camarilla_r3)
    camarilla_s3_array = np.full(len(df_1d), camarilla_s3)
    camarilla_s4_array = np.full(len(df_1d), camarilla_s4)
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_array)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_array)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_array)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_array)
    
    # Bullish engulfing: current candle closes above previous candle's high AND opens below previous candle's low
    bullish_engulfing = (close > high[np.arange(len(high))-1]) & (open_price < low[np.arange(len(low))-1])
    # Handle first element
    bullish_engulfing = np.insert(bullish_engulfing[1:], 0, False)
    
    # Bearish engulfing: current candle closes below previous candle's low AND opens above previous candle's high
    bearish_engulfing = (close < low[np.arange(len(low))-1]) & (open_price > high[np.arange(len(high))-1])
    # Handle first element
    bearish_engulfing = np.insert(bearish_engulfing[1:], 0, False)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion bounce conditions
        # Long when price touches or goes below S3/S4 with bullish engulfing
        long_setup = ((low[i] <= camarilla_s3_aligned[i]) or (low[i] <= camarilla_s4_aligned[i])) and bullish_engulfing[i]
        # Short when price touches or goes above R3/R4 with bearish engulfing
        short_setup = ((high[i] >= camarilla_r3_aligned[i]) or (high[i] >= camarilla_r4_aligned[i])) and bearish_engulfing[i]
        
        # Exit conditions: return to midpoint between S3/R3 or opposite touch
        camarilla_mid = (camarilla_s3_aligned[i] + camarilla_r3_aligned[i]) / 2
        long_exit = close[i] >= camarilla_mid
        short_exit = close[i] <= camarilla_mid
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

#!/usr/bin/env python3