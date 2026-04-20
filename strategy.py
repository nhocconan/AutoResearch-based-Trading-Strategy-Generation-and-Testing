#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(25) breakout with weekly trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; weekly EMA40 filters for higher timeframe trend alignment
# Volume > 1.6x 25-period average confirms institutional participation
# ATR-based stop loss manages risk via signal=0 when stop hit
# Designed for 12h timeframe with selective entries to avoid overtrading
# Target: 15-35 trades per year per symbol (60-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 40-period EMA on weekly timeframe for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate Donchian channels (25-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper and lower Donchian bands
    donchian_high = pd.Series(high).rolling(window=25, min_periods=25).max().values
    donchian_low = pd.Series(low).rolling(window=25, min_periods=25).min().values
    
    # Calculate ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    # Volume filter: volume > 1.6x 25-period average
    vol_ma = pd.Series(volume).rolling(window=25, min_periods=25).mean().values
    vol_filter = volume > (vol_ma * 1.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(80, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema40_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema40_1w_aligned[i]
        is_downtrend = close[i] < ema40_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + weekly uptrend + volume
            long_signal = (price > donchian_high[i]) and is_uptrend and has_volume
            
            # Short entry: price breaks below Donchian low + weekly downtrend + volume
            short_signal = (price < donchian_low[i]) and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.28
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or Donchian low break
            stop_loss = entry_price - 2.7 * atr[i]
            donchian_break = price < donchian_low[i]
            
            if stop_loss <= 0 or price <= stop_loss or donchian_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Short exit: stop loss or Donchian high break
            stop_loss = entry_price + 2.7 * atr[i]
            donchian_break = price > donchian_high[i]
            
            if stop_loss <= 0 or price >= stop_loss or donchian_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals

name = "12h_Donchian25_WeeklyTrendFilter_Volume_ATR"
timeframe = "12h"
leverage = 1.0