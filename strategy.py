#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band (20-period high) and close > 1d EMA34 (uptrend) with volume > 2.0x average.
Short when price breaks below Donchian lower band (20-period low) and close < 1d EMA34 (downtrend) with volume > 2.0x average.
Exit on opposite Donchian band break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear breakout levels, EMA34 filters medium-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets across both bull and bear regimes.
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
    
    # Align 1d EMA34 to 4h timeframe
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
        
        # Calculate Donchian bands (20-period) using only historical data
        lookback_start = max(0, i - 20)
        donchian_high = np.max(high[lookback_start:i])  # exclude current bar
        donchian_low = np.min(low[lookback_start:i])    # exclude current bar
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > donchian_high and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < donchian_low and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR trend reversal
                if (price < donchian_low or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR trend reversal
                if (price > donchian_high or price > ema34_val):
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