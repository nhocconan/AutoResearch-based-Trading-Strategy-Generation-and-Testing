#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA34 trend filter, volume spike confirmation, and ATR-based stoploss.
Targets 25-40 trades/year by requiring: 1) price breaks 20-period Donchian channel on 4h,
2) aligned with 1d EMA34 trend, 3) volume > 2.0x 20-period average (strong confluence),
4) initial stoploss at 2.0*ATR(14) from entry, trailing stop at highest/lowest since entry minus/plus 2.5*ATR.
Uses 4h timeframe to capture significant moves with controlled frequency. Donchian provides objective breakout levels.
Volume spike ensures institutional participation. EMA34 filter avoids counter-trend trades. ATR stop manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian(20) + volume MA(20) + ATR(14) + 1d EMA34(34)
    start_idx = max(20, 20, 14, 34) + 5
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with Donchian breakout, volume spike, and trend alignment
            # Long breakout: price breaks above Donchian upper band with uptrend and volume spike
            long_breakout = (curr_close > highest_20[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below Donchian lower band with downtrend and volume spike
            short_breakout = (curr_close < lowest_20[i]) and downtrend and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest since entry and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. Price reverses to Donchian lower band (mean reversion)
            # 2. Trailing stop hit: price drops below highest_since_entry - 2.5 * ATR
            # 3. Trend change (optional filter)
            donchian_lower = lowest_20[i]
            trailing_stop = highest_since_entry - 2.5 * atr[i]
            
            if curr_close < donchian_lower or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: update lowest since entry and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. Price reverses to Donchian upper band (mean reversion)
            # 2. Trailing stop hit: price rises above lowest_since_entry + 2.5 * ATR
            # 3. Trend change (optional filter)
            donchian_upper = highest_20[i]
            trailing_stop = lowest_since_entry + 2.5 * atr[i]
            
            if curr_close > donchian_upper or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0