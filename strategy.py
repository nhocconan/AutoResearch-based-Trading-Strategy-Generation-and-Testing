#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime
Hypothesis: Uses Camarilla pivot levels (R1, S1) from 1h timeframe for breakout entries, filtered by 4h EMA50 trend and 1d chop regime (CHOP < 61.8 = trending). Enters long when price breaks above 1h R1 AND 4h close > 4h EMA50 (uptrend) AND 1d not choppy. Enters short when price breaks below 1h S1 AND 4h close < 4h EMA50 (downtrend) AND 1d not choppy. Exits when price reverts to 1h close (mean reversion) OR trend breaks. Uses 1h timeframe with tight entries to avoid fee drag: target 15-37 trades/year. Camarilla levels provide high-probability breakout points, trend filter avoids counter-trend trades, and chop filter prevents whipsaws in ranging markets. Works in both bull and bear markets via 4h trend filter and 1d regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla pivot calculation
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate EMA50 on 4h close for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels for 1h timeframe
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    hl_range_1h = df_1h['high'].values - df_1h['low'].values
    camarilla_r1_1h = df_1h['close'].values + hl_range_1h * 1.1 / 12
    camarilla_s1_1h = df_1h['close'].values - hl_range_1h * 1.1 / 12
    
    # Align 1h indicators to 1h timeframe (no alignment needed as we're already in 1h)
    camarilla_r1_1h_aligned = camarilla_r1_1h  # already 1h resolution
    camarilla_s1_1h_aligned = camarilla_s1_1h  # already 1h resolution
    close_1h_aligned = df_1h['close'].values  # already 1h resolution
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness index on 1d timeframe
    # CHOP = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    tr = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:], 
                    np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr = np.maximum(tr, np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1]))
    tr = np.concatenate([[0], tr])  # align length
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / np.log10(14) / (max_high - min_low + 1e-10))
    chop_regime = chop > 61.8  # True = choppy/range, avoid breakouts
    chop_filter = ~chop_regime  # Only allow breakouts in trending markets
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period average (slightly relaxed from 2.0)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need 4h EMA50 (50), volume avg (20), chop (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_1h_aligned[i]) or np.isnan(camarilla_s1_1h_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_4h_val = ema_50_4h_aligned[i]
        r1_1h_val = camarilla_r1_1h_aligned[i]
        s1_1h_val = camarilla_s1_1h_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter_aligned[i] > 0.5  # convert to boolean
        close_1h_val = close_1h_aligned[i]
        
        if position == 0:
            # Look for entry: price breakout above R1 (long) or below S1 (short) with trend and regime filter
            # Long: price > R1 AND 4h uptrend AND volume confirmation AND not choppy
            long_condition = (close_val > r1_1h_val) and (close_val > ema_4h_val) and vol_conf and chop_ok
            # Short: price < S1 AND 4h downtrend AND volume confirmation AND not choppy
            short_condition = (close_val < s1_1h_val) and (close_val < ema_4h_val) and vol_conf and chop_ok
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 1h close (mean reversion) OR trend breaks
            exit_condition = (close_val <= close_1h_val) or (close_val < ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 1h close (mean reversion) OR trend breaks
            exit_condition = (close_val >= close_1h_val) or (close_val > ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime"
timeframe = "1h"
leverage = 1.0