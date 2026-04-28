#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR(14) volatility filter.
# Uses 4h primary timeframe for balanced trade frequency (~20-50/year expected).
# Donchian breakouts capture institutional price action with proven edge in crypto.
# 12h EMA50 filters for trend alignment, reducing counter-trend trades.
# ATR filter ensures breakouts occur during sufficient volatility, avoiding low-momentum false signals.
# Position size 0.25 for controlled risk. Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_Donchian20_12hEMA50_Trend_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > 0.5 * atr_ma_50  # ATR > 50% of its 50-bar MA
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA50 direction
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_20[i]
        short_breakout = close[i] < lowest_20[i]
        
        # Volatility filter
        vol_filter = volatility_filter[i]
        
        long_entry = price_above_ema and long_breakout and vol_filter
        short_entry = price_below_ema and short_breakout and vol_filter
        
        # Exit conditions: opposite Donchian level (10-period for faster exit)
        highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        
        long_exit = close[i] < lowest_10[i]  # Exit long at 10-period low
        short_exit = close[i] > highest_10[i]  # Exit short at 10-period high
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals