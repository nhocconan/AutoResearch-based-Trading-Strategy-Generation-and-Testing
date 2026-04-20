#!/usr/bin/env python3
"""
1d_1w_Donchian20_Breakout_Volume_Trend_v1
Concept: Weekly trend filter with daily Donchian breakout and volume confirmation.
- Long when price breaks above 20-day Donchian high with volume confirmation and above weekly EMA50
- Short when price breaks below 20-day Donchian low with volume confirmation and below weekly EMA50
- Exit when price returns to 10-day Donchian midpoint (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Weekly trend filter prevents counter-trend trades, volume confirms breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Weekly EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low with minimum periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        donchian_mid_val = donchian_mid[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(donchian_mid_val) or np.isnan(ema50_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and above weekly EMA50
            breakout_long = close_val > donchian_high_val
            vol_confirm = vol_ratio_val > 1.5  # Volume significantly above average
            
            if breakout_long and vol_confirm and close_val > ema50_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation and below weekly EMA50
            elif close_val < donchian_low_val and vol_confirm and close_val < ema50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint (mean reversion)
            if close_val <= donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint (mean reversion)
            if close_val >= donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals