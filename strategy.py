#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.inf
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 4h data for breakout levels (more responsive than daily)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    
    # 4h Donchian(15) breakout levels
    high_15 = pd.Series(df_4h['high'].values).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(df_4h['low'].values).rolling(window=15, min_periods=15).min().values
    high_15_aligned = align_htf_to_ltf(prices, df_4h, high_15)
    low_15_aligned = align_htf_to_ltf(prices, df_4h, low_15)
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_15_aligned[i]) or np.isnan(low_15_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
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
        
        # Volatility filter: only trade when ATR is above its 30-period median (avoid chop)
        if i >= 30:
            atr_med = np.median(atr_1d_aligned[i-29:i+1])
            vol_filter = atr_1d_aligned[i] > atr_med
        else:
            vol_filter = True  # Not enough data for median, allow trade
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Entry conditions: 
        # Long: break above 4h Donchian high with upward trend and volatility
        # Short: break below 4h Donchian low with downward trend and volatility
        long_breakout = close[i] > high_15_aligned[i]
        short_breakout = close[i] < low_15_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite Donchian level touch
        long_exit = (close[i] < low_15_aligned[i]) and position == 1
        short_exit = (close[i] > high_15_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
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
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_4hDonchian15_1dEMA34_VolatilityFilter"
timeframe = "12h"
leverage = 1.0