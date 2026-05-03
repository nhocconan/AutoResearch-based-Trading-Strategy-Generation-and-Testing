#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based stoploss.
# Long when price breaks above 4h Donchian upper channel + bull trend (close > 1d EMA34).
# Short when price breaks below 4h Donchian lower channel + bear trend (close < 1d EMA34).
# Uses ATR(14) for dynamic stoploss: exit long if price drops 2*ATR from entry, exit short if price rises 2*ATR from entry.
# Designed for 75-200 total trades over 4 years (19-50/year) with Sharpe > 0.5 on BTC/ETH/SOL.
# Works in bull via breakout continuation and in bear via short breakdowns with trend filter.

name = "4h_Donchian20_1dEMA34_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        donchian_upper = high_roll_max[i]
        donchian_lower = low_roll_min[i]
        ema_trend = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > donchian_upper
        short_breakout = close_val < donchian_lower
        
        # Entry logic
        if position == 0:
            if is_bull_trend and long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                max_high_since_entry = close_val
                min_low_since_entry = close_val
            elif is_bear_trend and short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                max_high_since_entry = close_val
                min_low_since_entry = close_val
        elif position == 1:
            # Update max high for trailing stop
            max_high_since_entry = max(max_high_since_entry, close_val)
            # ATR stoploss: exit if price drops 2*ATR from entry
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update min low for trailing stop
            min_low_since_entry = min(min_low_since_entry, close_val)
            # ATR stoploss: exit if price rises 2*ATR from entry
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals