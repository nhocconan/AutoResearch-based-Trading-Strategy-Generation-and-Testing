#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss
# - Long: Price breaks above Donchian(20) upper band + 1d volume > 1.3x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + 1d volume > 1.3x 20-period MA
# - Exit: ATR-based trailing stop (3x ATR from extreme) or opposite Donchian breakout
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# - Donchian breakouts capture strong momentum moves; volume confirmation filters weak breakouts
# - ATR trailing stop manages risk without look-ahead, works in both bull and bear markets

name = "4h_1d_donchian_volume_atr_stop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels for 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) for 4h (for stoploss)
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_stop = 0.0  # Trailing stop for long positions
    short_stop = 0.0  # Trailing stop for short positions
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for Donchian20 and ATR14)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h data
        close_price = close_4h[i]
        high_price = high_4h[i]
        low_price = low_4h[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_ma_current = volume_ma_aligned[i]
        atr_current = atr[i]
        
        # Volume spike condition: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d_current > 1.3 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + volume spike
            if (close_price > donchian_upper[i] and volume_spike):
                position = 1
                long_stop = close_price - 3.0 * atr_current  # Initial stop
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume spike
            elif (close_price < donchian_lower[i] and volume_spike):
                position = -1
                short_stop = close_price + 3.0 * atr_current  # Initial stop
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - manage trail stop and exit
            # Update trailing stop: only move up, never down
            long_stop = max(long_stop, close_price - 3.0 * atr_current)
            
            # Check stoploss or opposite signal
            if close_price < long_stop:  # Stoploss hit
                position = 0
                long_stop = 0.0
                signals[i] = 0.0
            elif close_price < donchian_lower[i]:  # Opposite Donchian breakout
                position = 0
                long_stop = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1, Short position - manage trail stop and exit
            # Update trailing stop: only move down, never up
            short_stop = min(short_stop, close_price + 3.0 * atr_current)
            
            # Check stoploss or opposite signal
            if close_price > short_stop:  # Stoploss hit
                position = 0
                short_stop = 0.0
                signals[i] = 0.0
            elif close_price > donchian_upper[i]:  # Opposite Donchian breakout
                position = 0
                short_stop = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals