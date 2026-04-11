#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long/short with 1d trend filter and volume confirmation.
# Enters long when price touches Camarilla L3 level with bullish 1d trend and expanding volume.
# Enters short when price touches Camarilla H3 level with bearish 1d trend and expanding volume.
# Uses tight stoploss at L4/H4 levels to manage risk.
# Designed for 20-40 trades/year on 4h timeframe with focus on mean reversion in range-bound markets.
# Works in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_H4 = prev_1d_close + 1.5 * (prev_1d_high - prev_1d_low)
    camarilla_H3 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low)
    camarilla_L3 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low)
    camarilla_L4 = prev_1d_close - 1.5 * (prev_1d_high - prev_1d_low)
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from index 1 to have previous bar data
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_bullish_trend = close[i] > ema_50_1d_aligned[i]
        is_bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla levels with volume and trend
        long_entry = (low[i] <= L3_aligned[i]) and vol_filter and is_bullish_trend
        short_entry = (high[i] >= H3_aligned[i]) and vol_filter and is_bearish_trend
        
        # Exit conditions: price reaches opposite Camarilla level or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when price reaches L4 (stop) or H3 (target) or trend turns bearish
            exit_long = (low[i] <= L4_aligned[i]) or (high[i] >= H3_aligned[i]) or not is_bullish_trend
        elif position == -1:
            # Exit short when price reaches H4 (stop) or L3 (target) or trend turns bullish
            exit_short = (high[i] >= H4_aligned[i]) or (low[i] <= L3_aligned[i]) or not is_bearish_trend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals