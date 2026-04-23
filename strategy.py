#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND 4h volume > 1.8x 20-period average volume.
Short when price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND 4h volume > 1.8x 20-period average volume.
Exit when price reaches Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~25-40 trades/year on 4h timeframe.
Camarilla levels calculated from prior completed 1d bar, ensuring no look-ahead bias.
Combines price structure (Camarilla pivots), trend filter (12h EMA50), and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior completed 1d bar (no look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed 1d bars
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed 1d bar (no look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    # Camarilla calculation: based on prior day's range
    rng = high_1d_shifted - low_1d_shifted
    pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3  # Pivot Point
    r3 = pp + (high_1d_shifted - low_1d_shifted) * 1.1 / 4  # Resistance 3
    s3 = pp - (high_1d_shifted - low_1d_shifted) * 1.1 / 4  # Support 3
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 4h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA20 and EMA50 need 20 and 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_50_12h_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        pp = pp_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish trend (12h close > EMA50) AND volume spike
            if close[i] > r3 and ema_50_12h_aligned[i] > ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND bearish trend (12h close < EMA50) AND volume spike
            elif close[i] < s3 and ema_50_12h_aligned[i] < ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price reaches Camarilla pivot point (PP)
            if position == 1 and close[i] >= pp:
                exit_signal = True
            elif position == -1 and close[i] <= pp:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0