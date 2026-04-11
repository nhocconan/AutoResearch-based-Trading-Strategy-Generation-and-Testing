#!/usr/bin/env python3
# 6h_1d_elder_ray_zone_v1
# Strategy: 6-period Elder Ray (Bull/Bear Power) with 1-day zone filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power via EMA13 deviation. Combining with 1-day high/low zones
# filters for institutional accumulation/distribution zones. Works in bull (buy power > 0 in uptrend zones)
# and bear (sell power < 0 in downtrend zones). Volume confirms institutional participation.
# Target: 50-150 total trades over 4 years = 12-37/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_zone_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Elder Ray components (13-period EMA)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1-day zone definitions (institutional accumulation/distribution zones)
    # Accumulation zone: below 1-day VWAP (institutional buying area)
    # Distribution zone: above 1-day VWAP (institutional selling area)
    # We'll use typical price approximation for VWAP: (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Define zones: below VWAP = accumulation (long bias), above VWAP = distribution (short bias)
    # Shift by 1 to use previous day's VWAP (known at 6h bar open)
    vwap_1d_prev = np.concatenate([np.array([vwap_1d[0]]), vwap_1d[:-1]])  # pad first value
    
    # Align 1-day VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_prev)
    
    # Zone signals: 1 = accumulation (long bias), -1 = distribution (short bias), 0 = neutral
    zone_long = close < vwap_1d_aligned  # price below VWAP = accumulation zone
    zone_short = close > vwap_1d_aligned  # price above VWAP = distribution zone
    
    # 20-period volume average for institutional participation confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Elder Ray signals with zone filter
        # Long: Bull Power > 0 (buying pressure) AND in accumulation zone AND volume confirmation
        long_signal = bull_power[i] > 0 and zone_long[i] and vol_confirm
        # Short: Bear Power < 0 (selling pressure) AND in distribution zone AND volume confirmation
        short_signal = bear_power[i] < 0 and zone_short[i] and vol_confirm
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Elder Ray signal or zone change
        elif position == 1 and (bear_power[i] < 0 or not zone_long[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] > 0 or not zone_short[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals