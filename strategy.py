#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day Donchian channel breakout with volume confirmation and 1-day EMA50 trend filter.
- Long when price breaks above 1-day Donchian high (20-period) + volume > 1.8x 20-period 12h volume MA + price above 1-day EMA50
- Short when price breaks below 1-day Donchian low (20-period) + volume > 1.8x 20-period 12h volume MA + price below 1-day EMA50
- Fixed position size 0.25 to manage drawdown
- Uses price channel breakout structure (works in ranging and trending markets) + volume confirmation + trend filter
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Donchian breakout captures volatility expansion, effective in both accumulation (bull) and distribution (bear) phases
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
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Donchian, close, and EMA50 trend filter (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1-day Donchian channel (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF indicators to primary timeframe (12h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_50_val = ema_50_aligned[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and 1-day EMA50 trend filter
            # Long: price breaks above 1-day Donchian high + volume spike + price above 1-day EMA50
            if price > donchian_high and vol > 1.8 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1-day Donchian low + volume spike + price below 1-day EMA50
            elif price < donchian_low and vol > 1.8 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1-day EMA50 (trend change) or opposite Donchian level
            if price < ema_50_val or price < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1-day EMA50 (trend change) or opposite Donchian level
            if price > ema_50_val or price > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_1dEMA50"
timeframe = "12h"
leverage = 1.0