#!/usr/bin/env python3
"""
1d_1w_WeeklyDonchianBreakout_VolumeTrend
Hypothesis: Weekly Donchian breakout (20-week) with volume confirmation and 1-week EMA trend filter.
Trades breakouts in trending markets (EMA20) and mean-reversion at channel bounds in ranging markets.
Designed for low trade frequency (target: 10-25 trades/year) to avoid fee drag while capturing multi-week trends.
Works in both bull and bear markets by adapting to trend (breakouts) and range (mean reversion) conditions.
"""

name = "1d_1w_WeeklyDonchianBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for Donchian channels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA20 for trend filter ---
    close_1w = df_1w['close']
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # --- Weekly Donchian Channels (20-period) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper channel: highest high of past 20 weeks
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of past 20 weeks
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily (Donchian levels are valid for the entire week)
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # --- Volume Spike Detection (2.0x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start after warmup
    start_idx = 60  # Need enough weekly data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
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
        
        # Determine trend based on price vs EMA20
        price_above_ema = close[i] > ema_20_1w_aligned[i]
        price_below_ema = close[i] < ema_20_1w_aligned[i]
        
        # Breakout signals (price breaks weekly Donchian with volume confirmation)
        # No minimum distance filter to allow natural breakouts
        long_breakout = (high[i] > upper_20_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < lower_20_aligned[i]) and vol_spike[i]
        
        # Mean reversion at channel bounds (price touches opposite band)
        # Only in non-trending conditions (price near EMA within 1.0%)
        near_ema = abs(close[i] - ema_20_1w_aligned[i]) / ema_20_1w_aligned[i] * 100 < 1.0
        long_reversion = (low[i] <= lower_20_aligned[i]) and near_ema and not vol_spike[i]
        short_reversion = (high[i] >= upper_20_aligned[i]) and near_ame and not vol_spike[i]
        
        if position == 0:
            # Enforce minimum 3-bar hold after entry (prevents immediate reversal)
            if bars_since_entry < 3:
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
                # Near EMA: allow mean reversion at channel bounds
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
                # Exit long: price touches lower band (support) or breaks below EMA
                exit_signal = (low[i] <= lower_20_aligned[i]) or (close[i] < ema_20_1w_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches upper band (resistance) or breaks above EMA
                exit_signal = (high[i] >= upper_20_aligned[i]) or (close[i] > ema_20_1w_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals