#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with volume confirmation and 1d EMA34 trend filter.
Long when price breaks above 4h Camarilla R3 AND volume > 1.5x 20-period average AND close > 1d EMA34.
Short when price breaks below 4h Camarilla S3 AND volume > 1.5x 20-period average AND close < 1d EMA34.
Exit when price retraces to 4h Camarilla pivot point (PP) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.20) to minimize fee drag and manage drawdown.
Designed for 1h timeframe targeting 15-37 trades/year per symbol (60-150 total over 4 years).
Focus on BTC and ETH as primary targets with volume and trend confirmation to filter false breakouts.
Works in both bull and bear markets: trend filter (1d EMA34) ensures alignment with higher timeframe direction,
while Camarilla levels provide institutional support/resistance for breakouts.
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
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla calculation: based on previous 4h bar's high, low, close
    camarilla_pp = (h_4h + l_4h + c_4h) / 3.0
    camarilla_range = h_4h - l_4h
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # vol MA needs 20, EMA needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        pp_val = camarilla_pp_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h Camarilla R3 AND volume spike AND above 1d EMA34 (uptrend)
            if price > r3_val and volume[i] > 1.5 * vol_ma_val and price > ema_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Price breaks below 4h Camarilla S3 AND volume spike AND below 1d EMA34 (downtrend)
            elif price < s3_val and volume[i] > 1.5 * vol_ma_val and price < ema_val:
                signals[i] = -0.20
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
            
            # Primary exit: Price retraces to 4h Camarilla pivot point
            if position == 1 and price <= pp_val:
                exit_signal = True
            elif position == -1 and price >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop_PPExit"
timeframe = "1h"
leverage = 1.0