#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_ChopFilter_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts with 12h EMA20 trend filter (price > EMA20 = uptrend) and chop regime filter (CHOP(14) < 61.8 = trending) to avoid whipsaw in ranging markets. Volume confirmation at 1.5x average reduces false breakouts. ATR trailing stop (2.0) manages risk. Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trends in both bull and bear markets via HTF regime alignment.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Previous 12h bar's high, low, close for Camarilla levels
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1 from 12h data
    camarilla_range_12h = prev_high_12h - prev_low_12h
    R1_12h = prev_close_12h + camarilla_range_12h * 1.0/12
    S1_12h = prev_close_12h - camarilla_range_12h * 1.0/12
    
    # Align 12h indicators to 4h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    # Chopiness Index (CHOP) on 4h - trending when < 61.8, ranging when > 61.8
    def calculate_chop(high, low, close, window=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop_14 = calculate_chop(high, low, close, 14)
    
    # Volume confirmation: 1.5x average volume (tighter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 4h)
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 12h EMA (20), volume MA (20), CHOP (14), 4h ATR (14)
    start_idx = max(20, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(R1_12h_aligned[i]) or 
            np.isnan(S1_12h_aligned[i]) or 
            np.isnan(chop_14[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_20_12h_val = ema_20_12h_aligned[i]
        R1_val = R1_12h_aligned[i]
        S1_val = S1_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        chop_val = chop_14[i]
        vol_ma_val = vol_ma[i]
        atr_14_4h_val = atr_14_4h[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        is_trending = chop_val < 61.8
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA20), volume spike, trending regime
            long_signal = (high_val > R1_val) and (close_val > ema_20_12h_val) and (volume_val > 1.5 * vol_ma_val) and is_trending
            # Short: break below S1, downtrend (close < EMA20), volume spike, trending regime
            short_signal = (low_val < S1_val) and (close_val < ema_20_12h_val) and (volume_val > 1.5 * vol_ma_val) and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_4h_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_4h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (close < EMA20) or chop too high (range)
            if (low_val < long_stop) or (close_val < ema_20_12h_val) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (close > EMA20) or chop too high (range)
            if (high_val > short_stop) or (close_val > ema_20_12h_val) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_ChopFilter_VolumeConfirm"
timeframe = "4h"
leverage = 1.0