#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_PowerTrend_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Elder Ray: Bull/Bear Power from Daily ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of daily close for power calculation
    close_series_1d = pd.Series(close_1d)
    ema13_1d = close_series_1d.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe (no extra delay needed for Elder Ray)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6h EMA Trend Filter ===
    close_series = pd.Series(prices['close'].values)
    ema34 = close_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema89 = close_series.ewm(span=89, min_periods=89, adjust=False).mean().values
    
    # === Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema34_val = ema34[i]
        ema89_val = ema89[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(ema34_val) or np.isnan(ema89_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (both bulls and bears agree on strength)
            #         with uptrend (EMA34 > EMA89) and volume confirmation
            if bull_val > 0 and bear_val < 0 and ema34_val > ema89_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (both agree on weakness)
            #        with downtrend (EMA34 < EMA89) and volume confirmation
            elif bear_val > 0 and bull_val < 0 and ema34_val < ema89_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Power divergence or trend breakdown
            if bull_val <= 0 or bear_val >= 0 or ema34_val < ema89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Power divergence or trend reversal
            if bear_val <= 0 or bull_val >= 0 or ema34_val > ema89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals