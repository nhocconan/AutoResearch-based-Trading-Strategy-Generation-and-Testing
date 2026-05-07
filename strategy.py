#!/usr/bin/env python3
name = "4h_Keltner_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Load daily data ONCE for trend filter and Keltner
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(20) for Keltner center and trend
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    upper_keltner = ema_20_1d + 2 * atr_1d
    lower_keltner = ema_20_1d - 2 * atr_1d
    
    # Align daily indicators to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume spike: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_20_aligned[i] > ema_20_aligned[i-1]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and daily downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner channel breakout with daily trend and volume confirmation
# - Keltner(20,2) uses ATR-based bands that adapt to volatility
# - Breakout above upper band with volume in daily uptrend = long opportunity
# - Breakdown below lower band with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Exit when price returns to EMA(20) or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Works in both bull and bear markets via daily trend filter
# - ATR-based bands provide better volatility adaptation than fixed percentage bands
# - Novel combination: Keltner breakout (4h) + trend (1d) + volume (4h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits