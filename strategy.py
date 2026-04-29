#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND close > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when Bear Power < 0 AND Bull Power > 0 AND close < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when Elder Power signals reverse (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid overtrading.
# Elder Ray measures bull/bear strength relative to EMA13, providing clear trend strength signals.
# Works in bull markets by capturing strong uptrends and in bear markets by shorting strong downtrends
# with weekly trend filter preventing counter-trend trades.

name = "6h_ElderRay_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Elder Ray calculation (needs high/low/close for EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray and EMA50 to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.5x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 50)  # volume MA, EMA13, EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening bullish momentum)
            if curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (weakening bearish momentum)
            if curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND close > 1w EMA50 AND volume confirmation
            if curr_bull > 0 and curr_bear < 0 and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND Bull Power > 0 AND close < 1w EMA50 AND volume confirmation
            elif curr_bear < 0 and curr_bull > 0 and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals