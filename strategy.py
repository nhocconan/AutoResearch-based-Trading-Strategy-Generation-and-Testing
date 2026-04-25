#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp_CHOPFilter
Hypothesis: 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike confirmation, and Choppiness Index regime filter (CHOP > 61.8 = range, only trade mean-reversion off pivots; CHOP < 38.2 = trend, only trade breakouts). Designed to reduce false signals in choppy markets and capture strong trends while minimizing overtrading (target: 12-37 trades/year). Works in bull via trend-following breakouts and in bear via mean-reversion in ranges.
"""

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
    
    # 1d data for EMA34 trend filter, Camarilla pivots, and volume average (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 0.5 * prev_range
    S1 = prev_close - 0.5 * prev_range
    R3 = prev_close + 1.5 * prev_range
    S3 = prev_close - 1.5 * prev_range
    
    # Align 1d pivot levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average (strict to reduce trade frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index (CHOP) - 14 period
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and CHOP (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Choppiness regime: CHOP > 61.8 = range, CHOP < 38.2 = trend
        chop_val = chop[i]
        is_range = chop_val > 61.8
        is_trend = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals with volume spike
            # In trend regime: trade breakouts
            # Long breakout: price breaks above R1 with uptrend and volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i] and is_trend
            # Short breakout: price breaks below S1 with downtrend and volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i] and is_trend
            
            # In range regime: trade mean reversion off extreme levels
            # Long mean reversion: price touches S3 and reverses up with volume spike
            long_mr = (curr_low <= S3_aligned[i]) and (curr_close > S3_aligned[i]) and volume_spike[i] and is_range
            # Short mean reversion: price touches R3 and reverses down with volume spike
            short_mr = (curr_high >= R3_aligned[i]) and (curr_close < R3_aligned[i]) and volume_spike[i] and is_range
            
            if long_breakout or long_mr:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout or short_mr:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry (using 12h ATR)
            # Calculate 12h ATR
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions based on regime
            elif is_range and curr_close > (R1_aligned[i] + S1_aligned[i]) / 2:  # Exit at midpoint in range
                signals[i] = 0.0
                position = 0
            elif is_trend and (curr_close < S1_aligned[i] or not uptrend):  # Exit on trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 12h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions based on regime
            elif is_range and curr_close < (R1_aligned[i] + S1_aligned[i]) / 2:  # Exit at midpoint in range
                signals[i] = 0.0
                position = 0
            elif is_trend and (curr_close > R1_aligned[i] or not downtrend):  # Exit on trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp_CHOPFilter"
timeframe = "12h"
leverage = 1.0