#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and trend filter.
# Uses 4h for signal direction (trend/structure), 1h only for precise entry timing.
# Designed to work in both bull and bear markets by requiring strong volume and trend alignment.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
name = "1h_4h_Donchian20_Breakout_VolumeTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper and lower bands using rolling window
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Trend Filter: 1h EMA50 > EMA200 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === Session Filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute session hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session (08-20 UTC)
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(ema50_val) or 
            np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume confirmation and uptrend
            if close_val > donchian_high_val and vol_ratio_val > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Donchian low with volume confirmation and downtrend
            elif close_val < donchian_low_val and vol_ratio_val > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low OR trend breaks down
            if close_val < donchian_low_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns above Donchian high OR trend breaks up
            if close_val > donchian_high_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals