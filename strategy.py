#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_VolumeSpike_KAMATrend
Hypothesis: Weekly Donchian channel breakout (20-period) with daily volume confirmation and KAMA trend filter captures institutional moves across bull and bear markets. Weekly timeframe reduces noise, daily volume confirms institutional participation, and KAMA adapts to trend changes. Targets 10-25 trades/year on 1d to minimize fee decay while capturing significant trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Donchian upper = max(high, lookback), lower = min(low, lookback)
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (only use after weekly bar closes)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily KAMA for trend filter (adaptive to market conditions)
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    price_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility > 0, price_change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily volume confirmation: volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian, KAMA, and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        kama_val = kama[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with volume spike and above KAMA
            if close[i] > upper and vol_spike_val and close[i] > kama_val:
                signals[i] = size
                position = 1
            # Short: break below weekly Donchian low with volume spike and below KAMA
            elif close[i] < lower and vol_spike_val and close[i] < kama_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below weekly Donchian low or price crosses below KAMA
            if close[i] < lower or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above weekly Donchian high or price crosses above KAMA
            if close[i] > upper or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_VolumeSpike_KAMATrend"
timeframe = "1d"
leverage = 1.0