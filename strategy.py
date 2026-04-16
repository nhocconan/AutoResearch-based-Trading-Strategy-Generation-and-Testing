#!/usr/bin/env python3
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
    
    # === 4h data for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA200 for trend filter
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1d data for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d ATR for volatility regime
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h Donchian breakout for entry timing ===
    highest_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = highest_20_4h
    donchian_lower_4h = lowest_20_4h
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema200_4h_val = ema200_4h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h Donchian lower OR volatility too high
            if (price < donchian_lower_4h_aligned[i]) or (atr_1d_val > 2.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h Donchian upper OR volatility too high
            if (price > donchian_upper_4h_aligned[i]) or (atr_1d_val > 2.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above 4h Donchian upper AND above 4h EMA200 (uptrend)
                if (price > donchian_upper_4h_aligned[i]) and (price > ema200_4h_val):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below 4h Donchian lower AND below 4h EMA200 (downtrend)
                elif (price < donchian_lower_4h_aligned[i]) and (price < ema200_4h_val):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA200_Donchian_Breakout_Session"
timeframe = "4h"
leverage = 1.0