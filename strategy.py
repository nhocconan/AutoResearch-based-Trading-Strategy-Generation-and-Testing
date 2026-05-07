#!/usr/bin/env python3
name = "4h_1d_Keltner_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(20) and ATR(10) for Keltner Channels
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
            np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
        )
    )
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA(20) ± 1.5 * ATR(10)
    upper_keltner = ema_20_1d + (atr_10_1d * 1.5)
    lower_keltner = ema_20_1d - (atr_10_1d * 1.5)
    
    # Align daily Keltner levels to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Wait for EMA(50) and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 1.8
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and daily downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_20_1d_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_20_1d_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner Channel breakout with daily trend and volume confirmation
# - Daily Keltner Channels (EMA20 ± 1.5*ATR10) act as dynamic support/resistance
# - Breakout above upper Keltner with volume in daily uptrend = long opportunity
# - Breakdown below lower Keltner with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to EMA(20) or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses daily Keltner Channels for better adaptation to volatility than fixed channels
# - Designed to work in BOTH bull and bear markets via trend filter
# - Avoids overtrading by requiring confluence of breakout, volume, and trend