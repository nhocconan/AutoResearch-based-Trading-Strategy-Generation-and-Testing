#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF context (as required)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR-based volatility filter (low volatility = range bound)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_1d < atr_ma_1d * 1.2  # Low volatility regime
    
    # Align daily ATR volatility filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Daily range for volatility breakout
    daily_range = high_1d - low_1d
    
    # Calculate 20-period SMA of daily range for breakout threshold
    range_ma = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    range_ma_aligned = align_htf_to_ltf(prices, df_1d, range_ma)
    
    # Volatility breakout signals: when daily range expands significantly
    volatility_expansion = daily_range > range_ma * 1.5
    volatility_expansion_aligned = align_htf_to_ltf(prices, df_1d, volatility_expansion)
    
    # 4-hour momentum confirmation (EMA crossover)
    ema_fast = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    momentum_up = ema_fast > ema_slow
    momentum_down = ema_fast < ema_slow
    
    # Time filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_filter_aligned[i]) or np.isnan(volatility_expansion_aligned[i]) or 
            np.isnan(range_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: low volatility environment (range bound)
        vol_cond = vol_filter_aligned[i]
        
        # Volatility expansion: significant increase in daily range
        vol_expansion = volatility_expansion_aligned[i]
        
        # Momentum confirmation
        mom_up = momentum_up[i]
        mom_down = momentum_down[i]
        
        # Entry conditions: 
        # Long: volatility expansion + upward momentum in low vol environment
        # Short: volatility expansion + downward momentum in low vol environment
        long_entry = vol_expansion and mom_up and vol_cond
        short_entry = vol_expansion and mom_down and vol_cond
        
        # Exit conditions: momentum reversal
        long_exit = not mom_up  # Exit long when momentum turns down
        short_exit = not mom_down  # Exit short when momentum turns up
        
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

name = "4h_VolatilityExpansion_Momentum_LowVol_Filter"
timeframe = "4h"
leverage = 1.0