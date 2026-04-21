#!/usr/bin/env python3
"""
Hypothesis: 4-hour price breaks above/below the 24-period Donchian channel with volume confirmation (1.5x 20-period 12h volume average) and 12-hour EMA50 trend filter. Long when price > upper Donchian + volume surge + 12h close > EMA50. Short when price < lower Donchian + volume surge + 12h close < EMA50. Exit when price returns to the opposite Donchian band or volume drops below average. Uses 0.25 position size to limit drawdown. Designed for low trade frequency (target 20-50/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA50 and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channel (24-period high/low for fewer signals)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=24, min_periods=24).max().values
    donchian_low = pd.Series(low_4h).rolling(window=24, min_periods=24).min().values
    
    # Align 12h indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 12h volume aligned)
        price_close = prices['close'].iloc[i]
        vol_12h_current = vol_12h_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume surge, 12h close > EMA50 (uptrend)
            if (price_close > donchian_high[i] and 
                vol_12h_current > 1.5 * vol_ma_20_aligned[i] and
                df_12h['close'].iloc[min(i//4, len(df_12h)-1)] > ema_50[min(i//4, len(df_12h)-1)]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume surge, 12h close < EMA50 (downtrend)
            elif (price_close < donchian_low[i] and 
                  vol_12h_current > 1.5 * vol_ma_20_aligned[i] and
                  df_12h['close'].iloc[min(i//4, len(df_12h)-1)] < ema_50[min(i//4, len(df_12h)-1)]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian band or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= lower Donchian or volume < average
                if (price_close <= donchian_low[i] or
                    vol_12h_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= upper Donchian or volume < average
                if (price_close >= donchian_high[i] or
                    vol_12h_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian24_Volume1.5x_EMA50_Trend"
timeframe = "4h"
leverage = 1.0