#!/usr/bin/env python3
"""
Experiment #4987: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with 1d weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) with volume confirmation (>1.5x average) capture strong momentum moves. Weekly pivots derived from 1d OHLC provide institutional reference points that work in both bull (breakouts with bias) and bear (breakdowns against bias) markets. Uses ATR(14) trailing stop (2.0x) to limit downside. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4987_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # Group by week (starting Monday) and aggregate
        df_1d_copy = df_1d.copy()
        df_1d_copy['week'] = pd.to_datetime(df_1d_copy['open_time']).dt.isocalendar().week
        df_1d_copy['year'] = pd.to_datetime(df_1d_copy['open_time']).dt.isocalendar().year
        
        # Weekly aggregation: week's open = first day's open, high = max(high), low = min(low), close = last day's close
        weekly_agg = df_1d_copy.groupby(['year', 'week']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        
        if len(weekly_agg) >= 2:
            # Use prior week's OHLC for current week's pivot (avoid look-ahead)
            weekly_agg['pivot'] = (weekly_agg['high'].shift(1) + weekly_agg['low'].shift(1) + weekly_agg['close'].shift(1)) / 3.0
            
            # Align weekly pivot to daily timeframe (each day gets prior week's pivot)
            # Create a mapping from date to week/year
            df_1d_copy['date_only'] = pd.to_datetime(df_1d_copy['open_time']).dt.date
            weekly_agg['date_start'] = pd.to_datetime(weekly_agg['year'].astype(str) + '-W' + 
                                                     weekly_agg['week'].astype(str) + '-1', format='%G-W%V-%u').dt.date
            
            # For each day, find the prior week's pivot
            pivot_values = []
            for idx, row in df_1d_copy.iterrows():
                current_date = row['date_only']
                # Find weeks that started before current date
                prior_weeks = weekly_agg[weekly_agg['date_start'] < current_date]
                if len(prior_weeks) > 0:
                    pivot_values.append(prior_weeks.iloc[-1]['pivot'])
                else:
                    pivot_values.append(np.nan)
            
            weekly_pivot_1d = np.array(pivot_values)
        else:
            weekly_pivot_1d = np.full(len(df_1d), np.nan)
    else:
        weekly_pivot_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF weekly pivot to 6h timeframe
    if len(weekly_pivot_1d) > 0:
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: price > pivot = bullish bias, price < pivot = bearish bias
        bullish_bias = price > weekly_pivot_aligned[i]
        bearish_bias = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions with pivot bias alignment
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals