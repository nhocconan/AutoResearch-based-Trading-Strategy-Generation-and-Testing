#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h breakout above/below daily Camarilla R1/S1 levels with 1d EMA34 trend filter and volume spike (>1.8x 20-bar MA). Designed for 20-50 trades/year (80-200 total over 4 years) to minimize fee drag. Uses discrete position sizing (0.25) and includes ATR-based stoploss (2.5x ATR) to control drawdown. Works in both bull and bear markets by following 1d trend while using Camarilla structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 1h data for better precision)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        atr_1h = np.full(n, 0.01)  # fallback
    else:
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        close_1h = df_1h['close'].values
        tr1 = np.maximum(high_1h[1:] - low_1h[1:], np.abs(high_1h[1:] - close_1h[:-1]))
        tr2 = np.maximum(np.abs(low_1h[1:] - close_1h[:-1]), tr1)
        tr = np.concatenate([[0], tr2])  # first tr = 0
        atr_1h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (tighter breakout for fewer trades)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.8x 20-period average (slightly looser for more trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema, 14 for atr)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_aligned[i] if 'atr_1h_aligned' in locals() else 0.01
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla level in trend direction with volume
        long_entry = (close_val > camarilla_r1_val) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_1d and vol_spike
        
        # Stoploss conditions: 2.5 * ATR from entry
        stop_long = position == 1 and close_val < (entry_price - 2.5 * atr_val)
        stop_short = position == -1 and close_val > (entry_price + 2.5 * atr_val)
        
        # Exit conditions: opposite Camarilla level touch OR trend reversal
        exit_long = (close_val < camarilla_s1_val) or not bullish_1d
        exit_short = (close_val > camarilla_r1_val) or not bearish_1d
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and (exit_long or stop_long):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif position == -1 and (exit_short or stop_short):
            signals[i] = 0.0
            position = -1
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0