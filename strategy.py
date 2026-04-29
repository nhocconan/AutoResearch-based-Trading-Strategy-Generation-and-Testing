#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and session filter (08-20 UTC)
# Long when close > R1 AND price > 4h EMA50 AND volume > 1.5x 20-bar avg AND hour in [8,20) UTC
# Short when close < S1 AND price < 4h EMA50 AND volume > 1.5x 20-bar avg AND hour in [8,20) UTC
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) OR ATR-based stoploss (2.0x ATR)
# Uses discrete position sizing (0.20) to minimize fee drag. Target: 15-37 trades/year on 1h.
# Camarilla levels provide institutional support/resistance. 4h EMA50 filters counter-trend moves.
# Volume spike confirms institutional participation. Session filter reduces noise trades.
# This strategy avoids overtrading by requiring confluence of 4 strong conditions.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours < 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss
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
    R1 = prev_typical + (prev_range * 1.1 / 12)
    S1 = prev_typical - (prev_range * 1.1 / 12)
    R2 = prev_typical + (prev_range * 1.1 / 6)
    S2 = prev_typical - (prev_range * 1.1 / 6)
    R3 = prev_typical + (prev_range * 1.1 / 4)
    S3 = prev_typical - (prev_range * 1.1 / 4)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # EMA50, volume MA, ATR all need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        sess_conf = in_session[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_4h_aligned[i]
        r1_level = R1[i]
        s1_level = S1[i]
        r2_level = R2[i]
        s2_level = S2[i]
        r3_level = R3[i]
        s3_level = S3[i]
        atr_val = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Check stoploss: close < entry_price - 2.0 * ATR
            if curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: close < S1 (opposite level)
            elif curr_close < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Check stoploss: close > entry_price + 2.0 * ATR
            if curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: close > R1 (opposite level)
            elif curr_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when close > R1 AND price > 4h EMA50 AND volume confirmation AND session
            if curr_close > r1_level and curr_close > ema_50 and vol_conf and sess_conf:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short when close < S1 AND price < 4h EMA50 AND volume confirmation AND session
            elif curr_close < s1_level and curr_close < ema_50 and vol_conf and sess_conf:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals