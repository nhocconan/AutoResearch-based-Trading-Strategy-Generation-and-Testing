# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
- Long: Close breaks above Donchian upper band (20-period high) + volume > 1.5x 20-period MA + 1d close > 1d EMA(50)
- Short: Close breaks below Donchian lower band (20-period low) + volume > 1.5x 20-period MA + 1d close < 1d EMA(50)
- Exit: Opposite Donchian break or 2x ATR stop
- Position size: 0.25 (25% of capital)
- Designed to work in both bull (breakouts) and bear (breakdowns) markets with volume filter reducing false signals.
"""

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) - not needed as prices is already 4h
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h ATR(14) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume ratio (current / 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donchian_high = high_roll[i]
        donchian_low = low_roll[i]
        trend_filter = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio[i]
        
        # EXIT LOGIC
        if position == 1:  # Long position
            # Exit: close below Donchian low or 2x ATR stop
            if price < donchian_low or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: close above Donchian high or 2x ATR stop
            if price > donchian_high or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # ENTRY LOGIC (only when flat)
        if position == 0:
            # LONG: break above Donchian high + volume + uptrend
            if price > donchian_high and vol_ratio_val > 1.5 and close_1d[i // 16] > trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: break below Donchian low + volume + downtrend
            elif price < donchian_low and vol_ratio_val > 1.5 and close_1d[i // 16] < trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # HOLD POSITION
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0