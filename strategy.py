#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR(14) for volatility filter
    high_low = df_1w['high'] - df_1w['low']
    high_close = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    low_close = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 6h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Weekly trend filter: price above/below weekly EMA50
        trend_filter_long = price > ema_50_1w_aligned[i]
        trend_filter_short = price < ema_50_1w_aligned[i]
        
        # Volatility filter: current 6h ATR should be less than 2x weekly ATR
        # Calculate 6h ATR(14)
        if i >= 14:
            hl = high[i] - low[i]
            hc = np.abs(high[i] - close[i-1])
            lc = np.abs(low[i] - close[i-1])
            tr_6h = max(hl, hc, lc)
            # Simplified: use current TR vs weekly ATR
            vol_filter = tr_6h < (2.0 * atr_1w_aligned[i])
        else:
            vol_filter = False
        
        # Volume filter: above average volume
        vol_filter = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian high + weekly uptrend + volume + volatility
            if (price > donchian_high[i] and 
                trend_filter_long and 
                vol_filter and 
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low + weekly downtrend + volume + volatility
            elif (price < donchian_low[i] and 
                  trend_filter_short and 
                  vol_filter and 
                  vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly trend turns down
            if price < donchian_low[i] or not trend_filter_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly trend turns up
            if price > donchian_high[i] or not trend_filter_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wATR_EMA_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0