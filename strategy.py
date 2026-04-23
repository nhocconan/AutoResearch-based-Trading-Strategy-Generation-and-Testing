#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band and close > 1d EMA34 (uptrend) with volume > 1.8x average.
Short when price breaks below Donchian lower band and close < 1d EMA34 (downtrend) with volume > 1.8x average.
Exit on opposite band break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian provides dynamic support/resistance, EMA34 filters medium-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels from previous 4h bar (avoid look-ahead)
    # Upper band: 20-period high, Lower band: 20-period low
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_4h = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_4h = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > upper_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < lower_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR trend reversal
                if (price < lower_val or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR trend reversal
                if (price > upper_val or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0