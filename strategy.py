#!/usr/bin/env python3
name = "6h_1w_1d_WaveTrend_Oscillator_v1"
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
    
    # Load weekly data ONCE before loop for WaveTrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate WaveTrend (WT) oscillator on weekly
    # WT1 = EMA(EMA(hlc3, n1), n2)
    # WT2 = SMA(WT1, n3)
    hlc3 = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    esa = pd.Series(hlc3).ewm(span=10, adjust=False, min_periods=10).mean()
    d = pd.Series(abs(hlc3 - esa)).ewm(span=10, adjust=False, min_periods=10).mean()
    ei = pd.Series(np.where(d != 0, (hlc3 - esa) / d, 0)).ewm(span=10, adjust=False, min_periods=10).mean()
    wt1 = pd.Series(ei).ewm(span=21, adjust=False, min_periods=21).mean()
    wt2 = pd.Series(wt1).rolling(window=4, min_periods=4).mean()
    
    wt1_values = wt1.values
    wt2_values = wt2.values
    
    # Align weekly WT to 6h
    wt1_aligned = align_htf_to_ltf(prices, df_1w, wt1_values)
    wt2_aligned = align_htf_to_ltf(prices, df_1w, wt2_values)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(wt1_aligned[i]) or np.isnan(wt2_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: WT crosses above -50 (oversold recovery) with volume and daily uptrend
            wt_cross_up = wt1_aligned[i] > wt2_aligned[i] and wt1_aligned[i-1] <= wt2_aligned[i-1]
            wt_oversold = wt1_aligned[i] < -50
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if wt_cross_up and wt_oversold and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: WT crosses below 50 (overbought rejection) with volume and daily downtrend
            elif wt1_aligned[i] < wt2_aligned[i] and wt1_aligned[i-1] >= wt2_aligned[i-1] and \
                 wt1_aligned[i] > 50 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: WT crosses below 0 or volume drops
            wt_cross_down = wt1_aligned[i] < wt2_aligned[i] and wt1_aligned[i-1] >= wt2_aligned[i-1]
            if wt_cross_down or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WT crosses above 0 or volume drops
            wt_cross_up = wt1_aligned[i] > wt2_aligned[i] and wt1_aligned[i-1] <= wt2_aligned[i-1]
            if wt_cross_up or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h WaveTrend oscillator from weekly + daily trend + volume confirmation
# - WaveTrend (WT) identifies overbought/oversold conditions on weekly timeframe
# - Long when WT crosses above -50 from oversold (< -50) with volume in daily uptrend
# - Short when WT crosses below 50 from overbought (> 50) with volume in daily downtrend
# - Volume spike (1.5x average) confirms institutional participation
# - Works in BOTH bull (buy oversold bounces in uptrend) and bear (sell overbought rejections in downtrend)
# - Exit when WT crosses zero line or volume weakens
# - Position size 0.25 targets ~30-80 trades/year, avoiding fee drag
# - Novel: WT oscillator not recently tried on 6h; combines weekly oscillator with daily trend
# - Uses actual weekly data (no resampling) via mtf_data for proper alignment
# - WT provides early reversal signals vs lagging indicators like RSI/MACD
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits