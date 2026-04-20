#!/usr/bin/env python3
# 1d_1w_Keltner_Channel_Breakout_Volume_Confirmation
# Hypothesis: On daily chart, use 20-period EMA with ATR(10) bands (Keltner Channel) to identify breakouts.
# Long when price closes above upper band with volume confirmation; short when price closes below lower band with volume confirmation.
# Weekly trend filter: only take long signals when price is above weekly EMA50, only short when below weekly EMA50.
# This strategy aims to capture strong trending moves while filtering counter-trend noise.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_1w_Keltner_Channel_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Keltner Channel on daily data
    kc_period = 20
    atr_period = 10
    kc_multiplier = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full_like(high, np.nan)
    if len(high) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(high)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # EMA for Keltner middle line
    ema_kc = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Keltner Bands
    upper_band = ema_kc + kc_multiplier * atr
    lower_band = ema_kc - kc_multiplier * atr
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, atr_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: price closes above upper band + volume confirmation + weekly uptrend
            if close[i] > upper_band[i] and vol_confirm and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band + volume confirmation + weekly downtrend
            elif close[i] < lower_band[i] and vol_confirm and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below middle line (EMA)
            if close[i] < ema_kc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above middle line (EMA)
            if close[i] > ema_kc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals