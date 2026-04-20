#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_VolumeTrend_v1
Concept: Weekly trend with daily Donchian breakout and volume confirmation.
- Long when weekly trend is up (price above weekly EMA50) and price breaks daily Donchian(20) high with volume spike
- Short when weekly trend is down (price below weekly EMA50) and price breaks daily Donchian(20) low with volume spike
- Exit when price returns to weekly EMA50 (trend reversal)
- Conservative sizing (0.25) to manage drawdown and reduce trade frequency
- Works in bull/bear: Weekly trend filter adapts, volume confirms genuine breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_VolumeTrend_v1"
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
    
    # === Weekly: EMA50 trend filter ===
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # === Daily: Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for weekly EMA50 and daily Donchian
    
    for i in range(start_idx, n):
        # Get values
        weekly_ema50_val = weekly_ema50_aligned[i]
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_ema50_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + Donchian breakout high + volume confirmation
            weekly_uptrend = close_val > weekly_ema50_val
            breakout_high = close_val > donchian_high_val
            vol_confirm = vol_ratio_val > 1.5  # Volume 50% above average
            
            if weekly_uptrend and breakout_high and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + Donchian breakout low + volume confirmation
            elif (not weekly_uptrend) and (close_val < donchian_low_val) and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly EMA50 (trend reversal)
            if close_val <= weekly_ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly EMA50 (trend reversal)
            if close_val >= weekly_ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals