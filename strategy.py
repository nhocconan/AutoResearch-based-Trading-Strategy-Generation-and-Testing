#!/usr/bin/env python3
name = "1d_Donchian20_Trend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

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
    
    # 1w high/low for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower bands (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 periods for Donchian + 34 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_upper = close[i] > upper_20_aligned[i]
        price_below_lower = close[i] < lower_20_aligned[i]
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper + above 1w EMA34 + volume spike
            if price_above_upper and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + below 1w EMA34 + volume spike
            elif price_below_lower and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below Donchian lower OR trend reverses
                if price_below_lower or price_below_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above Donchian upper OR trend reverses
                if price_above_upper or price_above_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals