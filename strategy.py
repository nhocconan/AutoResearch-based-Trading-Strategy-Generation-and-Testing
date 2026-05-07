#!/usr/bin/env python3
name = "6h_Keltner_Breakout_VolumeTrend_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Keltner Channels (20-period EMA ± 2*ATR)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20_1d + 2 * atr_14
    lower_keltner = ema_20_1d - 2 * atr_14
    
    # Align Keltner Channels to 6h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 4)  # Wait for EMA, ATR, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and 12h downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below upper Keltner or volume drops
            if close[i] < upper_keltner_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above lower Keltner or volume drops
            if close[i] > lower_keltner_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Keltner Breakout with 12h Trend and Volume Confirmation
# - Keltner Channels (20 EMA ± 2*ATR) from daily timeframe define volatility-adjusted bands
# - Breakout above upper Keltner with volume in 12h uptrend = long opportunity
# - Breakdown below lower Keltner with volume in 12h downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation and reduces false breakouts
# - Works in both bull (buy upper band breaks in uptrend) and bear (sell lower band breaks in downtrend)
# - Exit when price returns to upper/lower Keltner or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses ATR-based channels (not fixed %) for better adaptation to volatility regimes
# - 12h trend filter reduces whipsaws vs using same timeframe
# - Novel for 6h: Keltner breakouts not recently tested with volume+trend confirmation
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits