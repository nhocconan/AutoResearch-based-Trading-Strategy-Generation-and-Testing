#!/usr/bin/env python3
name = "4h_1d_Keltner_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtrader.mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Daily Keltner Channel (20, 2.0) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Typical Price and ATR calculation for Keltner
    tp = (prev_high + prev_low + prev_close) / 3
    atr = np.abs(prev_high - prev_low)  # Simplified ATR for daily
    
    # Keltner Channel: EMA(TP) ± 2*ATR
    ema_tp = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ma = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema_tp + (2 * atr_ma)
    lower_keltner = ema_tp - (2 * atr_ma)
    
    # Align daily Keltner to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_tp_aligned = align_htf_to_ltf(prices, df_1d, ema_tp)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 8-period average (2 days of 4h bars)
    vol_ma_8 = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 8)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma_8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_8[i] * 2.0
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and daily downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(TP) or volume drops
            if close[i] < ema_tp_aligned[i] or volume[i] < vol_ma_8[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(TP) or volume drops
            if close[i] > ema_tp_aligned[i] or volume[i] < vol_ma_8[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner Channel breakout with daily trend and volume confirmation
# - Daily Keltner Channel (EMA(TP) ± 2*ATR) identifies dynamic support/resistance
# - Breakout above upper Keltner with volume spike in daily uptrend = long opportunity
# - Breakdown below lower Keltner with volume spike in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Exit when price returns to EMA(TP) or volume weakens
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Uses daily Keltner channels for better adaptation to volatility changes
# - EMA(50) trend filter ensures alignment with higher timeframe momentum
# - Simpler than ATR-based channels with fewer parameters to optimize