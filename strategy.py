#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume confirmation
# Long when close > R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when close < S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Camarilla levels provide institutional support/resistance. EMA34 filters counter-trend moves.
# Volume spike confirms institutional participation. Works in both bull and bear via trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss (optional, using Camarilla exit instead)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous day's typical price for Camarilla calculation
    prev_typical = pd.Series(typical_price).shift(1).values
    # Calculate range
    daily_range = high - low
    prev_range = pd.Series(daily_range).shift(1).values
    
    # Camarilla levels
    R3 = prev_typical + (prev_range * 1.1 / 4)
    S3 = prev_typical - (prev_range * 1.1 / 4)
    R4 = prev_typical + (prev_range * 1.1 / 2)
    S4 = prev_typical - (prev_range * 1.1 / 2)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # EMA34 and volume MA need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_34 = ema_34_1d_aligned[i]
        r3_level = R3[i]
        s3_level = S3[i]
        r4_level = R4[i]
        s4_level = S4[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Check exit: close < S3 (opposite level)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Check exit: close > R3 (opposite level)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when close > R3 AND price > 1d EMA34 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when close < S3 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals