#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_VolumeTrend_v1
Concept: 4h Donchian channel breakout with volume confirmation and 12h trend filter.
- Long when price breaks above 4h Donchian high(20) with volume confirmation and above 12h EMA50
- Short when price breaks below 4h Donchian low(20) with volume confirmation and below 12h EMA50
- Exit when price returns to 4h Donchian midpoint (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Donchian adapts to volatility, volume confirms breakouts, 12h EMA filters countertrend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # === 12h: EMA50 trend filter ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 4h: Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low using rolling window
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema50_12h_val = ema50_12h_aligned[i]
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_val) or np.isnan(donch_high_val) or np.isnan(donch_low_val) or 
            np.isnan(donch_mid_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and above 12h EMA50
            breakout_long = close_val > donch_high_val
            vol_confirm = vol_ratio_val > 1.3  # Volume above average
            
            if breakout_long and vol_confirm and close_val > ema50_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation and below 12h EMA50
            elif close_val < donch_low_val and vol_confirm and close_val < ema50_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below Donchian midpoint (mean reversion)
            if close_val <= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above Donchian midpoint (mean reversion)
            if close_val >= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals