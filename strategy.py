# The core logic is a trend-following strategy that uses a 1-day pivot point (from the previous day) to establish key levels and a 6-hour EMA to determine the trend direction. Entries are taken when the price breaks above the pivot point (for longs) or below it (for shorts) with the trend confirmed by the EMA. The strategy uses a simple ATR-based stop loss for exits.
# The pivot point acts as a key support/resistance level, and a break of this level with trend confirmation can signal the start of a new move. This approach is designed to work in both bull and bear markets by trading in the direction of the 6-hour EMA.
# The strategy is designed to have a low trade frequency by requiring a clear break of the pivot level and trend confirmation, which should help mitigate the impact of transaction costs.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_Breakout_EMA_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE for pivot points (high, low, close of previous day)
    df_1d = get_htf_data(prices, '1d')
    # Get 6h data ONCE for EMA trend
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_1d) < 2 or len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate 1-day pivot point: (high + low + close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # 6h EMA34 for trend direction
    close_6h = df_6h['close'].values
    ema_34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_34_6h)
    
    # 6h ATR for exit (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot = pivot_1d_aligned[i]
        ema_trend = ema_34_6h_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(pivot) or np.isnan(ema_trend) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above pivot with uptrend (price > EMA)
            if current_close > pivot and current_close > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below pivot with downtrend (price < EMA)
            elif current_close < pivot and current_close < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below pivot or ATR stop loss
            if current_close < pivot:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above pivot or ATR stop loss
            if current_close > pivot:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals