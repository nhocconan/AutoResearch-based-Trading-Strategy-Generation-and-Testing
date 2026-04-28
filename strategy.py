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
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema40_1w = close_1w_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Volume filter: above average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_d[i]) if i < len(atr_d) else True) or np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA40
        trend_up = close[i] > ema40_1w_aligned[i]
        trend_down = close[i] < ema40_1w_aligned[i]
        
        # Calculate current 6-bar range for volatility breakout
        if i >= 5:
            high_6bar = np.max(high[i-5:i+1])
            low_6bar = np.min(low[i-5:i+1])
            range_6bar = high_6bar - low_6bar
            
            # Volatility breakout: break above/below 6-bar range with expansion
            upper_break = close[i] > high_6bar + 0.3 * atr_d[i] * (6/24)  # scale ATR to 6h
            lower_break = close[i] < low_6bar - 0.3 * atr_d[i] * (6/24)
            
            # Entry conditions: volatility breakout with trend and volume
            long_entry = upper_break and vol_filter and trend_up
            short_entry = lower_break and vol_filter and trend_down
        else:
            long_entry = False
            short_entry = False
        
        # Exit conditions: opposite 3-bar extreme touch
        if i >= 2:
            low_3bar = np.min(low[i-2:i+1])
            high_3bar = np.max(high[i-2:i+1])
            long_exit = close[i] < low_3bar and position == 1
            short_exit = close[i] > high_3bar and position == -1
        else:
            long_exit = False
            short_exit = False
        
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

name = "6h_VolatilityBreakout_WeeklyTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0