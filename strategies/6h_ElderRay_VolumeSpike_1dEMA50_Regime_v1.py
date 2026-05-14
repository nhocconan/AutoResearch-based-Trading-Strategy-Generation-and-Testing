#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume Spike + Regime Filter
# Bull Power = High - EMA13; Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND volume > 2.0x 20-bar avg AND price > 1d EMA50
# Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND volume > 2.0x 20-bar avg AND price < 1d EMA50
# Exit when momentum diverges (Bull Power < 0 for longs, Bear Power < 0 for shorts) OR price crosses EMA13
# Uses discrete position sizing (0.25) to reduce fee drag.
# Elder Ray captures institutional buying/selling pressure, volume confirmation ensures follow-through,
# 1d EMA50 filters counter-trend moves. Works in bull markets (strong bull power) and bear markets (strong bear power).

name = "6h_ElderRay_VolumeSpike_1dEMA50_Regime_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 50)  # volume MA, EMA13, and EMA50 alignment warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema13 = ema_13[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: bull power turns negative OR price crosses below EMA13
            if curr_bull < 0 or curr_close < curr_ema13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bear power turns negative OR price crosses above EMA13
            if curr_bear < 0 or curr_close > curr_ema13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when bull power > 0 AND bear power < 0 (bullish momentum) 
            # AND volume confirmation AND price > 1d EMA50
            if curr_bull > 0 and curr_bear < 0 and vol_conf and curr_close > curr_ema50:
                signals[i] = 0.25
                position = 1
            # Short when bear power > 0 AND bull power < 0 (bearish momentum)
            # AND volume confirmation AND price < 1d EMA50
            elif curr_bear > 0 and curr_bull < 0 and vol_conf and curr_close < curr_ema50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals