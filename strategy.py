#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND 1d volume > 2.0x 20-period average volume.
Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND 1d volume > 2.0x 20-period average volume.
Exit when price reaches Camarilla PP (pivot point) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-35 trades/year on 4h timeframe.
Camarilla levels provide intraday structure that works in both bull and bear markets when combined with trend and volume filters.
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
    
    # Calculate 1d OHLC for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Camarilla levels (based on previous day's OHLC)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2, PP = (H+L+C)/3
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    camarilla_pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average (20-period) for spike filter
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
    start_idx = max(34, 20)  # EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        
        # Current 1d values (use last value of 1d arrays, aligned to current 4h bar)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        vol_1d_val = vol_1d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish trend (close > EMA34) AND volume spike
            if close[i] > r3_val and close_1d_val > ema_val and vol_1d_val > 2.0 * vol_ma_1d_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND bearish trend (close < EMA34) AND volume spike
            elif close[i] < s3_val and close_1d_val < ema_val and vol_1d_val > 2.0 * vol_ma_1d_val:
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
            
            # Primary exit: Price reaches Camarilla PP (pivot point)
            if position == 1 and close[i] >= pp_val:
                exit_signal = True
            elif position == -1 and close[i] <= pp_val:
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

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0