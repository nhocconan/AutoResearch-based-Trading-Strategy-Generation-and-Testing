#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d close > 1d EMA50 AND 12h volume > 1.8x 20-period average volume.
Short when price breaks below Camarilla S3 AND 1d close < 1d EMA50 AND 12h volume > 1.8x 20-period average volume.
Exit when price reaches Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~12-37 trades/year on 12h timeframe.
Combines price structure (Camarilla pivot levels), trend filter (1d EMA50), and volume confirmation for robustness across bull/bear regimes.
Camarilla levels are calculated from prior 1d OHLC, ensuring no look-ahead bias.
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
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # 1d arrays for Camarilla calculation (use prior completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Camarilla levels from prior 1d OHLC (H, L, C)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, PP = (H+L+C)/3
    camarilla_R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_PP = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 12h timeframe (using prior completed 1d bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 12h trailing stop calculation
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
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_PP_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_50_1d_aligned[i]
        r3 = camarilla_R3_aligned[i]
        s3 = camarilla_S3_aligned[i]
        pp = camarilla_PP_aligned[i]
        
        # Current 1d close value (use last value of 1d arrays, aligned to current 12h bar)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_val = close_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish trend (close > EMA50) AND volume spike
            if close[i] > r3 and close_1d_val > ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND bearish trend (close < EMA50) AND volume spike
            elif close[i] < s3 and close_1d_val < ema_val and volume[i] > 1.8 * vol_ma_val:
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

name = "12H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0