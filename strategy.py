#!/usr/bin/env python3
"""
6h Camarilla H3L3 Breakout + 1d Weekly Pivot Direction + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance on 6h timeframe.
Breakouts aligned with weekly pivot direction (from 1d HTF) and volume confirmation capture
strong momentum moves while avoiding false breakouts in choppy markets. Weekly pivot adds
structural bias that works in both bull (buying above weekly pivot) and bear (selling below)
regimes. Discrete sizing (0.25) targets ~75-150 trades over 4 years to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from previous day
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Calculate weekly pivot from 1d data (using prior week's OHLC)
    # Need to group by week - use ISO week numbering
    df_1d_copy = df_1d.copy()
    df_1d_copy['year_week'] = df_1d_copy.index.isocalendar().year * 100 + df_1d_copy.index.isocalendar().week
    weekly_agg = df_1d_copy.groupby('year_week').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly_agg) < 2:
        return np.zeros(n)
    
    # Weekly pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_high = weekly_agg['high'].shift(1).values
    weekly_low = weekly_agg['low'].shift(1).values
    weekly_close = weekly_agg['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Map weekly pivot to daily index (forward fill)
    weekly_pivot_series = pd.Series(weekly_pivot, index=weekly_agg.index[1:])  # skip first NaN
    weekly_pivot_daily = weekly_pivot_series.reindex(df_1d.index[1:], method='ffill')
    weekly_pivot_daily = weekly_pivot_daily.reindex(df_1d.index)  # align with original df_1d
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_daily.values, additional_delay_bars=0)
    
    # Calculate ATR for stop loss (using 14 periods on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for ATR (14) and HTF data alignment
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions at Camarilla H3/L3 levels
        bullish_breakout = curr_close > camarilla_h3_val
        bearish_breakout = curr_close < camarilla_l3_val
        
        # Weekly pivot direction filter
        # Long only if price above weekly pivot, short only if below
        long_pivot_filter = curr_close > weekly_pivot_val
        short_pivot_filter = curr_close < weekly_pivot_val
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or pivot rejection
                elif curr_close < camarilla_l3_val or curr_close < weekly_pivot_val:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or pivot rejection
                elif curr_close > camarilla_h3_val or curr_close > weekly_pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + pivot direction + volume
        if position == 0:
            # Long: break above H3 AND price above weekly pivot AND volume spike
            long_condition = bullish_breakout and long_pivot_filter and volume_spike
            # Short: break below L3 AND price below weekly pivot AND volume spike
            short_condition = bearish_breakout and short_pivot_filter and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dWeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0