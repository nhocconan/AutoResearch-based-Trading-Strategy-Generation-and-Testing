#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) with volume spike confirmation and 1-day EMA trend filter on 12h timeframe.
Trades breakouts in trending markets (EMA34) and mean-reversion at pivot levels in ranging markets.
Designed for low trade frequency (target: 20-40 trades/year) to avoid fee drag while capturing high-probability moves.
Works in both bull and bear markets by adapting to trend (breakouts) and range (mean reversion) conditions.
"""

name = "12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Daily Camarilla Pivot Levels (R3, S3) ---
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4)  # R3
    s3 = pivot - (range_1d * 1.1 / 4)  # S3
    
    # Align to 12h (Camarilla levels are valid for the entire day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- Volume Spike Detection (2.5x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.5 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Determine trend based on price vs EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        # Distance from EMA as percentage
        ema_distance_pct = abs(close[i] - ema_34_1d_aligned[i]) / ema_34_1d_aligned[i] * 100
        
        # Breakout signals (price crosses R3/S3 with volume spike and sufficient EMA separation)
        # Requires price to be >1.5% away from EMA to avoid whipsaw
        long_breakout = (high[i] > r3_aligned[i]) and vol_spike[i] and (ema_distance_pct > 1.5)
        short_breakout = (low[i] < s3_aligned[i]) and vol_spike[i] and (ema_distance_pct > 1.5)
        
        # Mean reversion at pivot levels (price touches S3/R3 without breakout)
        # Only in non-trending conditions (price near EMA within 0.5%)
        near_ema = ema_distance_pct < 0.5
        long_reversion = (low[i] <= s3_aligned[i]) and near_ema and not vol_spike[i]
        short_reversion = (high[i] >= r3_aligned[i]) and near_ema and not vol_spike[i]
        
        if position == 0:
            # Enforce minimum 2-bar hold after entry (prevents immediate reversal)
            if bars_since_entry < 2:
                signals[i] = 0.0
                continue
                
            if price_above_ema:
                # Uptrend: favor long breakouts, avoid shorts
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
            elif price_below_ema:
                # Downtrend: favor short breakouts, avoid longs
                if short_breakout:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
            else:
                # Near EMA: allow mean reversion at pivot levels
                if long_reversion:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                elif short_reversion:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S3 (support) or breaks below EMA
                exit_signal = (low[i] <= s3_aligned[i]) or (close[i] < ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R3 (resistance) or breaks above EMA
                exit_signal = (high[i] >= r3_aligned[i]) or (close[i] > ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals