#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily close for calculations
    close_1d = df_1d['close'].values
    
    # Calculate daily return for volatility
    daily_return = np.diff(close_1d, prepend=close_1d[0]) / close_1d
    daily_vol = pd.Series(daily_return).rolling(window=10, min_periods=10).std().values
    
    # Calculate volatility percentile (20-day lookback)
    vol_percentile = pd.Series(daily_vol).rolling(window=20, min_periods=10).apply(
        lambda x: np.percentile(x, 50) if len(x) > 0 else 50, raw=False).values
    
    # Current volatility vs median (normalized)
    vol_ratio = daily_vol / (pd.Series(daily_vol).rolling(window=20, min_periods=10).median().values + 1e-10)
    
    # Align volatility ratio to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Low volatility filter (below median)
    low_vol_filter = vol_ratio_aligned < 1.0
    
    # Hour filter: 0-6 UTC (Asian session - typically lower volatility)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    asian_session = (hours >= 0) & (hours < 6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(vol_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade Asian session (0-6 UTC)
        if not asian_session[i]:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Low volatility filter
        vol_filter = low_vol_filter[i]
        
        if not vol_filter:
            # High volatility: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: price above previous day's close with low volatility in Asian session
        # Short: price below previous day's close with low volatility in Asian session
        long_entry = (close[i] > close_1d[i]) and vol_filter
        short_entry = (close[i] < close_1d[i]) and vol_filter
        
        # Exit conditions: price returns to previous day's close or volatility increases
        long_exit = (close[i] <= close_1d[i]) or (not vol_filter)
        short_exit = (close[i] >= close_1d[i]) or (not vol_filter)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "4h_AsianSession_LowVol_MeanReversion"
timeframe = "4h"
leverage = 1.0