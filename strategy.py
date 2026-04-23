#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band and close > 1d EMA34 (uptrend) with volume > 1.8x average.
Short when price breaks below Donchian lower band and close < 1d EMA34 (downtrend) with volume > 1.8x average.
Exit on opposite Donchian band break or trend reversal. Uses 12h timeframe targeting 50-150 total trades over 4 years.
Donchian channels provide robust trend-following structure, EMA34 filters medium-term trend, volume confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in both bull and bear markets.
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
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Donchian channels for 12h timeframe using 20-period lookback
        if i >= 20:
            lookback_start = i - 20
            upper_band = np.max(high[lookback_start:i])  # highest high of last 20 bars (excludes current)
            lower_band = np.min(low[lookback_start:i])   # lowest low of last 20 bars (excludes current)
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > upper_band and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < lower_band and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR trend reversal
                if (price < lower_band or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR trend reversal
                if (price > upper_band or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0