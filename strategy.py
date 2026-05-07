#!/usr/bin/env python3
name = "6h_1d_CCI_Trend_Reversion_v1"
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
    
    # Load daily data ONCE before loop for CCI and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily CCI(20)
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = tp_1d.rolling(window=20, min_periods=20).mean()
    mad = tp_1d.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    cci_1d = cci_1d.values
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI oversold (< -100) in daily uptrend with volume confirmation
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if cci_1d_aligned[i] < -100 and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: CCI overbought (> 100) in daily downtrend with volume confirmation
            elif cci_1d_aligned[i] > 100 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI returns to neutral or volume drops
            if cci_1d_aligned[i] > -50 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI returns to neutral or volume drops
            if cci_1d_aligned[i] < 50 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s CCI mean reversion with daily trend filter and volume confirmation
# - Daily CCI(20) identifies overbought/oversold conditions (>100/< -100)
# - Trade only in direction of daily EMA(50) trend to avoid counter-trend whipsaws
# - Long when daily CCI < -100 (oversold) + daily uptrend + volume spike
# - Short when daily CCI > 100 (overbought) + daily downtrend + volume spike
# - Volume confirmation (1.8x average) filters false signals
# - Exit when CCI returns toward neutral (-50/50) or volume weakens
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Novel: CCI on higher timeframe with trend filter not recently tried on 6h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits