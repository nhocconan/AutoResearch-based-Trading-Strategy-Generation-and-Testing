#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Donchian breakout + volume confirmation + trend filter
# Works in bull (breakouts capture momentum) and bear (mean reversion via trend filter)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
name = "12h_1d_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate upper and lower bands from previous 20 days
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper_band = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_band = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 12h timeframe (use previous day's levels)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h: EMA 34 for trend filter ===
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        ema34_val = ema34[i]
        vol_ratio_val = vol_ratio[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(vol_ratio_val) or 
            np.isnan(upper_val) or np.isnan(lower_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with volume confirmation and uptrend
            if close_val > upper_val and vol_ratio_val > 1.5 and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with volume confirmation and downtrend
            elif close_val < lower_val and vol_ratio_val > 1.5 and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below lower band or trend turns down
            if close_val < lower_val or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above upper band or trend turns up
            if close_val > upper_val or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals