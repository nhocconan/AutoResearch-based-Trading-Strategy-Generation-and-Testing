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
    
    # Get weekly data for trend filter and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.inf
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Get daily data for Donchian channel (15-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Daily Donchian(15) breakout levels
    high_15 = pd.Series(df_1d['high'].values).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(df_1d['low'].values).rolling(window=15, min_periods=15).min().values
    high_15_aligned = align_htf_to_ltf(prices, df_1d, high_15)
    low_15_aligned = align_htf_to_ltf(prices, df_1d, low_15)
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_15_aligned[i]) or np.isnan(low_15_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_1w_aligned[i])):
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
        
        # Volatility filter: only trade when ATR is above its 50-period median (avoid chop)
        if i >= 50:
            atr_ma = np.median(atr_1w_aligned[i-49:i+1])
            vol_filter = atr_1w_aligned[i] > atr_ma
        else:
            vol_filter = True  # Not enough data for median, allow trade
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: break above daily Donchian high with upward trend and volatility
        # Short: break below daily Donchian low with downward trend and volatility
        long_breakout = close[i] > high_15_aligned[i]
        short_breakout = close[i] < low_15_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite Donchian level touch
        long_exit = (close[i] < low_15_aligned[i]) and position == 1
        short_exit = (close[i] > high_15_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
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

name = "1d_Donchian15_1wEMA20_VolatilityFilter"
timeframe = "1d"
leverage = 1.0