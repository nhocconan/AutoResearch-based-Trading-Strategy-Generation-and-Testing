#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper (20) and close > 1d EMA50 (uptrend) with volume > 1.8x average.
Short when price breaks below 4h Donchian lower (20) and close < 1d EMA50 (downtrend) with volume > 1.8x average.
Exit on opposite Donchian break or trend reversal. Uses 1h for timing, 4h/1d for direction.
Session filter 08-20 UTC to avoid low-liquidity hours. Target 60-150 trades over 4 years.
Position size 0.20 discrete levels to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = prices.index.hour
    
    # Load 4h data for Donchian channel - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donch_hi = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe
    donch_hi_aligned = align_htf_to_ltf(prices, df_4h, donch_hi)
    donch_lo_aligned = align_htf_to_ltf(prices, df_4h, donch_lo)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_hi_val = donch_hi_aligned[i]
        donch_lo_val = donch_lo_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > donch_hi_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below 4h Donchian lower AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < donch_lo_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 4h Donchian lower OR trend reversal
                if (price < donch_lo_val or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 4h Donchian upper OR trend reversal
                if (price > donch_hi_val or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0