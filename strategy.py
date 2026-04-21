#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and 1d trend filter capture sustained moves in both bull and bear markets. The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume > 1.5x 20-period average confirms breakout strength. Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA50 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Calculate daily EMA50
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.full(n, np.nan)
    for i in range(20, n):
        volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema50 = ema50_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long breakout: price above Donchian upper + above daily EMA50 + volume
            if price > upper and price > ema50 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below Donchian lower + below daily EMA50 + volume
            elif price < lower and price < ema50 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower (failed breakout/reversal)
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper (failed breakdown/reversal)
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0