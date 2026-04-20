#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day ATR filter + volume confirmation
# Breakouts above 4h Donchian high/low signal momentum continuation
# 1-day ATR filter ensures volatility is elevated (ATR > 1.5x 20-day ATR average)
# Volume confirmation requires volume > 1.5x 20-period average
# Trend filter: price > 1-day EMA50 for longs, price < 1-day EMA50 for shorts
# Designed to capture strong momentum moves in both bull and bear markets
# Target: 20-50 total trades per year (80-200 over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period ATR on daily timeframe for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf  # First value has no previous close
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(atr20_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volatility filter: current ATR > 1.5x 20-day ATR average
        # We don't have 4h ATR directly, so use price range as proxy
        current_range = high[i] - low[i]
        vol_condition = current_range > (atr20_1d_aligned[i] * 1.5)
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + bull trend + vol + volume
            long_signal = False
            if is_bull and vol_condition and has_volume:
                if price > donchian_high[i]:
                    long_signal = True
            
            # Enter short: price breaks below Donchian low + bear trend + vol + volume
            short_signal = False
            if is_bear and vol_condition and has_volume:
                if price < donchian_low[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR bear trend
            exit_signal = False
            if price < donchian_low[i] or not is_bull:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR bull trend
            exit_signal = False
            if price > donchian_high[i] or not is_bear:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_TrendVolFilter_Volume"
timeframe = "4h"
leverage = 1.0