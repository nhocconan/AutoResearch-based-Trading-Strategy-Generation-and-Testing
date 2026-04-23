#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and close > 1w EMA50 with volume > 2.0x average.
Short when price breaks below lower Donchian channel and close < 1w EMA50 with volume > 2.0x average.
Exit on opposite Donchian break or trend reversal.
Donchian channels provide clear breakout levels that work in both trending and ranging markets.
1w EMA50 filters long-term trend, volume confirmation ensures breakout legitimacy.
Designed for 1d timeframe targeting 30-100 total trades over 4 years with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 1d data (20-period)
    lookback = 20
    upper_donchian = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_donchian = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        upper_donchian_val = upper_donchian[i]
        lower_donchian_val = lower_donchian[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > upper_donchian_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < lower_donchian_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR trend reversal
                if (price < lower_donchian_val or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR trend reversal
                if (price > upper_donchian_val or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0