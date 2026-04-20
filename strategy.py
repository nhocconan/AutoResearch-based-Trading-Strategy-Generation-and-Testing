#!/usr/bin/env python3
"""
1d_1w_KAMA_PriceChannel_Trend_v1
Concept: Daily KAMA trend combined with weekly price channel breakout and volume confirmation.
- Uses daily KAMA (Adaptive Moving Average) to determine trend direction
- Uses weekly Donchian channel (20-period) for breakout signals
- Requires volume confirmation (volume > 1.5x 20-period average)
- Long when price breaks above weekly upper channel AND daily KAMA is rising
- Short when price breaks below weekly lower channel AND daily KAMA is falling
- Exit when price crosses daily KAMA (trend reversal)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: KAMA adapts to volatility, price channels capture breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_PriceChannel_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for price channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: Donchian channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Daily: KAMA trend indicator ===
    close = prices['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama[i]
        kama_prev = kama[i-1] if i > 0 else kama_val
        close_val = close[i]
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(kama_prev) or np.isnan(close_val) or 
            np.isnan(upper_channel) or np.isnan(lower_channel) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly upper channel with volume confirmation and KAMA rising
            breakout_long = close_val > upper_channel
            vol_confirm = vol_ratio_val > 1.5  # Volume above average
            kama_rising = kama_val > kama_prev
            
            if breakout_long and vol_confirm and kama_rising:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly lower channel with volume confirmation and KAMA falling
            elif close_val < lower_channel and vol_confirm and kama_val < kama_prev:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below daily KAMA (trend reversal)
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above daily KAMA (trend reversal)
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals