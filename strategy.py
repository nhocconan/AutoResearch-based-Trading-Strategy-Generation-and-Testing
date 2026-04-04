#!/usr/bin/env python3
"""
Experiment #5099: 6h Donchian(20) Breakout + 12h/1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction from 1d timeframe capture institutional order flow. 
Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. 
Weekly pivot direction provides structural bias: above weekly pivot = long bias, below = short bias. 
Designed for 12-37 trades/year on 6h timeframe to minimize fee drag. Works in bull markets (breakouts with upward pivot bias) 
and bear markets (breakdowns with downward pivot bias). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5099_6h_donchian20_12h_1d_weekly_pivot_vol_v1"
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
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # Resample logic: group by week (starting Monday) and aggregate
        df_1d_copy = df_1d.copy()
        df_1d_copy['week_start'] = df_1d_copy.index.to_series().dt.to_period('W').dt.start_time
        weekly = df_1d_copy.groupby('week_start').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index(drop=True)
        
        # Calculate pivot points for each week: P = (H + L + C)/3
        # S1 = 2*P - H, R1 = 2*P - L
        # S2 = P - (H - L), R2 = P + (H - L)
        # S3 = H - 2*(H - P), R3 = L + 2*(P - L)
        weekly_high = weekly['high'].values
        weekly_low = weekly['low'].values
        weekly_close = weekly['close'].values
        
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        
        weekly_r3 = weekly_low + 2.0 * (weekly_pivot - weekly_low)  # R3 = L + 2*(P - L)
        weekly_s3 = weekly_high - 2.0 * (weekly_high - weekly_pivot)  # S3 = H - 2*(H - P)
        
        # Align weekly levels to daily frequency (each daily bar gets prior week's levels)
        # Create array of same length as df_1d, filled with NaN
        weekly_r3_daily = np.full(len(df_1d), np.nan)
        weekly_s3_daily = np.full(len(df_1d), np.nan)
        
        # For each week, fill all daily bars in that week with that week's R3/S3
        week_start_times = weekly['week_start'].values
        for week_idx, week_start in enumerate(week_start_times):
            # Find daily bars belonging to this week
            week_end = week_start + pd.Timedelta(days=7)
            mask = (df_1d.index >= week_start) & (df_1d.index < week_end)
            if np.any(mask):
                weekly_r3_daily[mask] = weekly_r3[week_idx]
                weekly_s3_daily[mask] = weekly_s3[week_idx]
        
        # Now align to 6h timeframe
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3_daily)
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3_daily)
    else:
        weekly_r3_aligned = np.full(n, np.nan)
        weekly_s3_aligned = np.full(n, np.nan)
    
    # Precompute HTF: 12h data for trend filter (optional)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        close_12h = df_12h['close'].values
        sma_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
        sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h)
    else:
        sma_12h_aligned = np.full(n, np.nan)
    
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
            np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Determine weekly pivot bias: price above weekly pivot = long bias, below = short bias
        # We approximate weekly pivot as midpoint between S3 and R3 for simplicity
        weekly_pivot_approx = (weekly_r3_aligned[i] + weekly_s3_aligned[i]) / 2.0
        pivot_bias_long = price > weekly_pivot_approx
        pivot_bias_short = price < weekly_pivot_approx
        
        # Donchian breakout conditions with weekly pivot bias filter
        # Long: Donchian breakout above + price > weekly pivot bias (bullish structure)
        # Short: Donchian breakdown below + price < weekly pivot bias (bearish structure)
        breakout_long = (price >= high_roll[i]) and pivot_bias_long and vol_confirm
        breakout_short = (price <= low_roll[i]) and pivot_bias_short and vol_confirm
        
        # Optional: Additional 12h trend filter (require price > 12h SMA for long, < for short)
        if not np.isnan(sma_12h_aligned[i]):
            breakout_long = breakout_long and (price > sma_12h_aligned[i])
            breakout_short = breakout_short and (price < sma_12h_aligned[i])
        
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