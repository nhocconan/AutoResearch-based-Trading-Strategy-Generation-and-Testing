#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) AND close > 1d EMA34 AND volume > 1.8x average.
Short when price breaks below lower Donchian(20) AND close < 1d EMA34 AND volume > 1.8x average.
Exit when price reverses to middle of Donchian channel OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 25-40 trades/year per symbol.
Works in bull markets via breakouts and bear markets via short breakdowns with trend filter.
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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: break above upper Donchian AND price > 1d EMA34 AND volume spike
            if (price > upper and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian AND price < 1d EMA34 AND volume spike
            elif (price < lower and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle of Donchian OR volume drops below average
                if (price <= mid or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle of Donchian OR volume drops below average
                if (price >= mid or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0