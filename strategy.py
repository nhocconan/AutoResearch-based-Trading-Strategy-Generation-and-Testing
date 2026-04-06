#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d Volatility Filter + ATR Stoploss
Hypothesis: Donchian breakouts on 12h capture multi-day momentum with low trade frequency. 
Volatility filter (ATR ratio) ensures we only trade during normal volatility, avoiding chop. 
ATR stoploss limits drawdown. Designed for 50-150 total trades over 4 years.
Works in bull/bear by only taking breakouts in direction of 1d trend (via price > EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1dvolfilter_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 50-period EMA on 1d for trend filter
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 49) / 51
    
    # 14-period ATR on 1d for volatility filter and stoploss
    atr_14 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        tr = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        atr_14[0] = np.nan
        if len(tr) > 0:
            atr_14[1] = tr[0]
            for i in range(2, len(atr_14)):
                atr_14[i] = (tr[i-1] * 13 + atr_14[i-1]) / 14
    
    # ATR ratio: current ATR / 50-period average ATR (volatility filter)
    atr_ma_50 = np.full_like(atr_14, np.nan)
    if len(atr_14) >= 50:
        # Find first valid ATR
        first_valid = 0
        while first_valid < len(atr_14) and np.isnan(atr_14[first_valid]):
            first_valid += 1
        if first_valid < len(atr_14):
            for i in range(first_valid + 49, len(atr_14)):
                start_idx = i - 49
                valid_slice = atr_14[start_idx:i+1]
                # Only use non-nan values
                valid_vals = valid_slice[~np.isnan(valid_slice)]
                if len(valid_vals) >= 20:  # Require minimum samples
                    atr_ma_50[i] = np.mean(valid_vals)
    
    # Align 1d indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR ratio is between 0.5 and 2.0
        # Avoids extremely low volatility (chop) and extremely high volatility (panic)
        atr_ratio = atr_14_aligned[i] / atr_ma_50_aligned[i] if atr_ma_50_aligned[i] > 0 else 1.0
        vol_filter = 0.5 <= atr_ratio <= 2.0
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against trend
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] < lowest_low or 
                not uptrend or
                close[i] < entry_price - 2.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against trend
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] > highest_high or 
                not downtrend or
                close[i] > entry_price + 2.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volatility filter + trend alignment
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and vol_filter and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and vol_filter and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals