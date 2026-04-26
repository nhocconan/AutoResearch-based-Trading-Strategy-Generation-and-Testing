#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Only trade breakouts in direction of 1d trend to avoid counter-trend whipsaw.
Volume spike confirms institutional interest. Designed for BTC/ETH with discrete sizing (0.25)
to limit fee drag. Target: 75-200 total trades over 4 years (19-50/year).
Uses ATR-based trailing stop (signal=0 when price < highest_high - 2.5*ATR for longs,
or price > lowest_low + 2.5*ATR for shorts) to manage risk without look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate ATR(20) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Track highest high since entry for trailing stop (longs)
    # Track lowest low since entry for trailing stop (shorts)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for Donchian/ATR)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Update trailing stop levels
        if position == 1:  # Long position
            highest_since_entry[i] = max(highest_since_entry[i-1] if i > start_idx else high[i], high[i])
        elif position == -1:  # Short position
            lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > start_idx else low[i], low[i])
        else:
            # Reset tracking when flat
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            vol_ma_20_val = volume[i]  # fallback
        else:
            vol_ma_20_val = vol_ma_20[i]
        volume_spike = volume[i] > 1.5 * vol_ma_20_val
        
        # Breakout conditions
        bullish_breakout = close[i] > highest_high[i]
        bearish_breakout = close[i] < lowest_low[i]
        
        # Trailing stop conditions
        long_stop = False
        short_stop = False
        if position == 1 and not np.isnan(highest_since_entry[i]):
            long_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
        elif position == -1 and not np.isnan(lowest_since_entry[i]):
            short_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
        
        # Exit on stop
        if long_stop or short_stop:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic
        if htf_trend[i] == 1:  # Uptrend on 1d
            if bullish_breakout and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            if bearish_breakout and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0