#!/usr/bin/env python3
"""
Experiment #5131: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot-derived trend 
(from 1d timeframe) capture strong momentum with institutional participation. 
Weekly pivot levels (calculated from prior week's OHLC on 1d data) provide structural 
support/resistance that works in both bull and bear markets. Volume > 1.5x average 
confirms genuine breakout. ATR(14) trailing stop (2.0x) manages risk. 
Target: 12-37 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5131_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Trend ===
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC
        # We need to group daily data into weeks
        # Create a DataFrame with date index for resampling
        df_1d_df = pd.DataFrame({
            'open': df_1d['open'],
            'high': df_1d['high'],
            'low': df_1d['low'],
            'close': df_1d['close']
        }, index=pd.to_datetime(df_1d.index))  # Assuming df_1d has datetime index
        
        # Resample to weekly OHLC (starting Monday)
        weekly = df_1d_df.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(weekly) >= 1:
            # Calculate weekly pivot points: P = (H+L+C)/3
            weekly_high = weekly['high'].values
            weekly_low = weekly['low'].values
            weekly_close = weekly['close'].values
            
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            
            # Trend: price above weekly pivot = uptrend, below = downtrend
            # We need to align this to 1d timeframe first
            # Map each daily bar to its weekly pivot value
            weekly_pivot_series = pd.Series(weekly_pivot, index=weekly.index)
            # Forward fill to get pivot value for each day in the week
            daily_pivot = weekly_pivot_series.reindex(df_1d_df.index, method='ffill')
            
            # Align to 6h timeframe
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot.values)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
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
        
        # Donchian breakout conditions with weekly pivot trend filter
        # Long: Donchian breakout above + price > weekly pivot (uptrend)
        # Short: Donchian breakdown below + price < weekly pivot (downtrend)
        breakout_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
        
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