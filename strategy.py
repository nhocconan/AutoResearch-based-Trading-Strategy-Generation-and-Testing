#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h trend filter + volume confirmation
# Long when price breaks above 6h Donchian upper band (20) with volume > 1.5x 20-bar average AND 12h close > 12h open (bullish candle)
# Short when price breaks below 6h Donchian lower band (20) with volume > 1.5x 20-bar average AND 12h close < 12h open (bearish candle)
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR, short exit when price > lowest_low_since_entry + 2.0 * ATR
# Donchian provides clear structure, 12h candle direction filters for higher-timeframe bias, volume confirms conviction.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee drag.

name = "6h_Donchian20_12hCandleDir_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data ONCE before loop for trend filter (bullish/bearish candle)
    df_12h = get_htf_data(prices, '12h')
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    bullish_12h = close_12h > open_12h  # True for bullish 12h candle
    bearish_12h = close_12h < open_12h  # True for bearish 12h candle
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 20) + 1  # Donchian(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band with volume spike AND bullish 12h candle
            if (close[i] > highest_high[i] and 
                volume_spike[i] and bullish_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Donchian lower band with volume spike AND bearish 12h candle
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and bearish_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals