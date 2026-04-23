#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with 1d EMA trend filter and volume spike confirmation.
Long when price breaks above 20-bar Donchian high AND close > 1d EMA50 (uptrend) AND volume > 2.0x 20-bar average.
Short when price breaks below 20-bar Donchian low AND close < 1d EMA50 (downtrend) AND volume > 2.0x 20-bar average.
Donchian channels provide objective breakout levels, EMA50 filters trend direction, volume spike confirms conviction.
Designed for 6h timeframe to achieve 50-150 total trades over 4 years with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donchian_high[i]
        donch_low_val = donchian_low[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > donch_high_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < donch_low_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low OR price breaks below 1d EMA50 (trend reversal)
                if price < donch_low_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high OR price breaks above 1d EMA50 (trend reversal)
                if price > donch_high_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0