#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike confirmation, and choppy market regime filter.
Targets 20-40 trades/year by requiring: 1) price breaks R1/S1 levels, 2) aligned with 1d EMA trend, 3) volume > 2.0x 20-period average, 4) choppiness index < 61.8 (non-choppy market).
Designed for low turnover and high edge in both bull/bear markets via trend alignment, institutional volume confirmation, and regime filtering to avoid whipsaws.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter and Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla pivots
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + 1.1 * prev_range * (1.0/12.0)  # R1 = C + 1.1*(HL/12)
    S1 = prev_close - 1.1 * prev_range * (1.0/12.0)  # S1 = C - 1.1*(HL/12)
    
    # Align 1d pivot levels and EMA to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (volume spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter (to avoid whipsaws in sideways markets)
    # CHOP = 100 * log10(sum(ATR over n) / (log10(n) * (max(high) - min(low)) over n))
    # We use a simplified version: CHOP < 61.8 indicates trending market (good for breakouts)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Choppiness Index: CHOP = 100 * log10(sum_atr_14 / range_14) / log10(14)
    # Avoid division by zero and log of zero
    chop_ratio = np.where((range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_atr_14)), 
                          sum_atr_14 / range_14, np.nan)
    chop = np.where((chop_ratio > 0) & (~np.isnan(chop_ratio)), 
                    100 * np.log10(chop_ratio) / np.log10(14), 50.0)  # default to 50 if invalid
    chop_filter = chop < 61.8  # Non-choppy/trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and 1d pivots (2) and chop calculation (14+14=28)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
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
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and regime filter
            # Long breakout: price breaks above R1 with uptrend, volume confirmation, and non-choppy market
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i] and chop_filter[i]
            # Short breakout: price breaks below S1 with downtrend, volume confirmation, and non-choppy market
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i] and chop_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry (using 4h ATR)
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes or market becomes choppy
            elif curr_close < S1_aligned[i] or not uptrend or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry (using 4h ATR)
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes or market becomes choppy
            elif curr_close > R1_aligned[i] or not downtrend or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0