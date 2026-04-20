#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Keltner_Breakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Keltner Channel (based on 20-period EMA and 2x ATR) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period EMA of close
    close_series_1d = pd.Series(close_1d)
    ema20_1d = close_series_1d.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # True Range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low to avoid look-ahead
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Keltner Bands
    upper_keltner = ema20_1d + 2 * atr_1d
    lower_keltner = ema20_1d - 2 * atr_1d
    
    # Align to 12h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === Volume Trend Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h Trend Filter: EMA50 > EMA200 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        upper_keltner_val = upper_keltner_aligned[i]
        lower_keltner_val = lower_keltner_aligned[i]
        ema20_val = ema20_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(upper_keltner_val) or 
            np.isnan(lower_keltner_val) or np.isnan(ema20_val) or 
            np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Keltner band with volume confirmation and uptrend
            if close_val > upper_keltner_val and vol_ratio_val > 1.8 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Keltner band with volume confirmation and downtrend
            elif close_val < lower_keltner_val and vol_ratio_val > 1.8 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA20 OR trend breaks down
            if close_val < ema20_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA20 OR trend breaks up
            if close_val > ema20_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals