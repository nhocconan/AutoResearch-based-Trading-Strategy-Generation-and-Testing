#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with daily trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper band, daily EMA50 trend is up, and volume > 1.5x average.
Short when price breaks below Donchian(20) lower band, daily EMA50 trend is down, and volume > 1.5x average.
Exit when price crosses back through the Donchian midpoint or volume drops below average.
Uses Donchian channels for breakout detection, daily EMA for trend filter, and volume for confirmation.
Designed for 15-25 trades/year to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, daily EMA50 up, volume surge
            if (price_high > donchian_high[i] and 
                ema_50_aligned[i] > ema_50_aligned[max(0, i-1)] and
                vol_1d[i] > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, daily EMA50 down, volume surge
            elif (price_low < donchian_low[i] and 
                  ema_50_aligned[i] < ema_50_aligned[max(0, i-1)] and
                  vol_1d[i] > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian midpoint or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price below Donchian midpoint OR volume < average
                if price_close < donchian_mid[i]:
                    exit_signal = True
                elif vol_1d[i] < vol_ma_20_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price above Donchian midpoint OR volume < average
                if price_close > donchian_mid[i]:
                    exit_signal = True
                elif vol_1d[i] < vol_ma_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_DailyEMA50_Trend_Volume1.5x"
timeframe = "12h"
leverage = 1.0