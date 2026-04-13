#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 12h trend filter
    # Long when Williams %R < -80 (oversold) and price > 12h EMA50 (uptrend)
    # Short when Williams %R > -20 (overbought) and price < 12h EMA50 (downtrend)
    # Exit when Williams %R crosses -50 (mean reversion completion)
    # Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
    # Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    williams_r = -100 * ((highest_high - close) / hl_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (williams_r[i] < -80) and (close[i] > ema_50_12h_aligned[i]) and (position != 1)
        short_entry = (williams_r[i] > -20) and (close[i] < ema_50_12h_aligned[i]) and (position != -1)
        
        # Exit conditions (mean reversion completion)
        exit_long = (position == 1 and williams_r[i] > -50)
        exit_short = (position == -1 and williams_r[i] < -50)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_12h_williamsr_ema_trend_filter_v1"
timeframe = "6h"
leverage = 1.0