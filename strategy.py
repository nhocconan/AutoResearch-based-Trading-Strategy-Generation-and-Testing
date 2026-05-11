#!/usr/bin/env python3
"""
12h_Volume_Weighted_Pullback_Strategy
Hypothesis: In trending markets (ADX > 25), pullbacks to the VWAP on volume spikes offer high-probability entries. In ranging markets (ADX < 25), fade extreme deviations from VWAP with volume confirmation. Uses 1d trend filter and volume spike confirmation to reduce false signals. Designed for 12h timeframe to target 50-150 total trades over 4 years.
"""

name = "12h_Volume_Weighted_Pullback_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (ADX) and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d ADX for trend detection (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- 12h VWAP calculation ---
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    vwap_num = np.cumsum(typical_price * volume_12h)
    vwap_den = np.cumsum(volume_12h)
    vwap = vwap_num / vwap_den
    
    # --- 12h Standard Deviation of price from VWAP (for deviation bands) ---
    price_dev = typical_price - vwap
    vwap_std = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    # --- 12h Volume Average for confirmation ---
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ADX and VWAP std
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from entry)
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30 if position == 1 else -0.30
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_12h_aligned[i] < 25
        is_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 12h average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        # Price deviation from VWAP in standard deviations
        if vwap_std[i] > 0:
            dev_sd = (typical_price[i] - vwap[i]) / vwap_std[i]
        else:
            dev_sd = 0
        
        if position == 0:
            # Look for entries based on regime
            if is_range and vol_confirm:
                # Mean reversion: fade extreme deviations from VWAP
                if dev_sd > 2.0:  # Price significantly above VWAP
                    signals[i] = -0.30  # short
                    position = -1
                    entry_price = close_12h[i]
                elif dev_sd < -2.0:  # Price significantly below VWAP
                    signals[i] = 0.30   # long
                    position = 1
                    entry_price = close_12h[i]
            elif is_trend and vol_confirm:
                # Trend following: pullback to VWAP in direction of trend
                if dev_sd < -0.5 and dev_sd > -2.0:  # Mild pullback below VWAP
                    signals[i] = 0.30  # long
                    position = 1
                    entry_price = close_12h[i]
                elif dev_sd > 0.5 and dev_sd < 2.0:  # Mild pullback above VWAP
                    signals[i] = -0.30  # short
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit when price returns to VWAP
                    if dev_sd >= -0.5:  # Price back to or above VWAP
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: price moves further away from VWAP
                    elif dev_sd < -2.5:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.30
                else:  # is_trend
                    # In trend, trail with VWAP or stop at extreme deviation
                    if dev_sd > 0.5:  # Price back above VWAP
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: extreme adverse deviation
                    elif dev_sd < -2.5:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.30
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit when price returns to VWAP
                    if dev_sd <= 0.5:  # Price back to or below VWAP
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: price moves further away from VWAP
                    elif dev_sd > 2.5:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.30
                else:  # is_trend
                    # In trend, trail with VWAP or stop at extreme deviation
                    if dev_sd < -0.5:  # Price back below VWAP
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: extreme adverse deviation
                    elif dev_sd > 2.5:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.30
    
    return signals