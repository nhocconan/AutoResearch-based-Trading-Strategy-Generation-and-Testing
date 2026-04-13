#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ATR-based stoploss
    # Long when: price breaks above Donchian(20) high AND volume > 2.0x 20-period avg volume
    # Short when: price breaks below Donchian(20) low AND volume > 2.0x 20-period avg volume
    # Exit: ATR trailing stop (long: price < highest_high_since_entry - 2.5*ATR; short: price > lowest_low_since_entry + 2.5*ATR)
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Volume filter reduces false breakouts; ATR stop manages risk in volatile markets.
    # Works in bull/bear via Donchian structure providing objective breakout levels.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels on 12h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0  # first bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track extreme prices for trailing stop
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Update trailing stop extremes
        if position == 1:  # long position
            if i == 100 or position == 0:  # new entry or just entered
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
        elif position == -1:  # short position
            if i == 100 or position == 0:  # new entry or just entered
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
        else:  # flat
            highest_since_entry[i] = np.nan
            lowest_since_entry[i] = np.nan
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # ATR trailing stop conditions
        stop_long = False
        stop_short = False
        if position == 1 and not np.isnan(highest_since_entry[i]):
            stop_long = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
        elif position == -1 and not np.isnan(lowest_since_entry[i]):
            stop_short = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
        
        # Entry conditions
        long_entry = long_breakout and vol_ok and position != 1
        short_entry = short_breakout and vol_ok and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
            highest_since_entry[i] = high[i]  # reset tracking
        elif short_entry:
            position = -1
            signals[i] = -position_size
            lowest_since_entry[i] = low[i]  # reset tracking
        elif position == 1 and (stop_long or close[i] < donchian_mid[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (stop_short or close[i] > donchian_mid[i]):
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_donchian_volume_atr_stop_v1"
timeframe = "12h"
leverage = 1.0