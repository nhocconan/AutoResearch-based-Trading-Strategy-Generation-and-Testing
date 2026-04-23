#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 1w Camarilla R3 AND price > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below 1w Camarilla S3 AND price < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price retraces to 1w Camarilla pivot point or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.30) for balanced risk/return.
Designed for 1d timeframe targeting ~7-25 trades/year per symbol (30-100 total over 4 years).
Focus on BTC and ETH as primary targets with SOL as secondary confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla levels: R3, S3, pivot (PP)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: based on previous week's range
    camarilla_pp = (h_1w + l_1w + c_1w) / 3.0
    camarilla_r3 = camarilla_pp + (h_1w - l_1w) * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - (h_1w - l_1w) * 1.1 / 4.0
    
    # Align 1w Camarilla levels to 1d timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla needs 1, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla R3 AND price > 1w EMA50 AND volume spike
            if (price > r3_val and price > ema_50_val and volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1w Camarilla S3 AND price < 1w EMA50 AND volume spike
            elif (price < s3_val and price < ema_50_val and volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1w Camarilla pivot point
            if position == 1 and price <= pp_val:
                exit_signal = True
            elif position == -1 and price >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0