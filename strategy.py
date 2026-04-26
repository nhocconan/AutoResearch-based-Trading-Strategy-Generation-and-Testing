#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_ATRStop_v1
Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with volume confirmation and EMA50 trend filter. Uses ATR-based stoploss and discrete position sizing (0.25) to limit fee drag. Designed to work in both bull and bear markets via trend filter and volatility-based stops. Targets 20-50 trades/year to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 4h, Donchian(20), ATR(14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        atr_val = atr[i]
        vol_filter = volume_filter[i]
        
        # Trend filter
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume
            long_signal = (close_val > donch_high_val) and \
                          uptrend and \
                          vol_filter
            
            # Short: break below Donchian low with downtrend and volume
            short_signal = (close_val < donch_low_val) and \
                           downtrend and \
                           vol_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR-based stoploss: exit if price drops 2.5 * ATR from highest
            if close_val < (highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR-based stoploss: exit if price rises 2.5 * ATR from lowest
            if close_val > (lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0