#!/usr/bin/env python3
"""
Hypothesis: 4-hour price crosses above/below 20-period Donchian channel with volume confirmation and 1-day EMA34 trend filter.
Long when price breaks above upper Donchian(20) with volume > 1.5x average and 1d close > EMA34.
Short when price breaks below lower Donchian(20) with volume > 1.5x average and 1d close < EMA34.
Exit when price returns to the opposite Donchian band or volume drops below average.
Donchian channels capture breakout momentum, volume confirms breakout strength,
and EMA34 filters for trend alignment to avoid counter-trend trades in chop.
Target: 20-50 trades/year for low fee drag and robust performance in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA34 and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 1d volume aligned)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume surge, 1d close > EMA34 (uptrend)
            if (price_close > donchian_high[i] and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                close_1d[np.searchsorted(df_1d.index, prices['index'][i], side='right') - 1] > ema_34[np.searchsorted(df_1d.index, prices['index'][i], side='right') - 1]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume surge, 1d close < EMA34 (downtrend)
            elif (price_close < donchian_low[i] and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  close_1d[np.searchsorted(df_1d.index, prices['index'][i], side='right') - 1] < ema_34[np.searchsorted(df_1d.index, prices['index'][i], side='right') - 1]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian band or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= lower Donchian or volume < average
                if (price_close <= donchian_low[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= upper Donchian or volume < average
                if (price_close >= donchian_high[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_EMA34_Trend"
timeframe = "4h"
leverage = 1.0