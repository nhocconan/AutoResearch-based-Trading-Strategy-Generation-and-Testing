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
    
    # Get daily data for ATR calculation and position sizing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter and stop sizing
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # ATR(14) - Wilder's smoothing
    atr_d = np.zeros_like(tr)
    atr_d[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_d[i] = (atr_d[i-1] * 13 + tr[i]) / 14
    
    atr_d_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: above 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy low-vol periods)
        vol_filter = atr_d_aligned[i] > (close[i] * 0.005)
        
        # Volume filter: above average volume
        vol_filter = vol_filter and (volume[i] > vol_ma[i] * 1.5)
        
        # Trend filter: price relative to weekly EMA50
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions: 
        # Long: momentum break above weekly EMA50 with volume and volatility
        # Short: momentum break below weekly EMA50 with volume and volatility
        long_entry = trend_up and vol_filter and (close[i] > close[i-1])
        short_entry = trend_down and vol_filter and (close[i] < close[i-1])
        
        # Exit conditions: 
        # Long exit: price breaks below weekly EMA50 or loss of momentum
        long_exit = (close[i] < ema50_1w_aligned[i]) or (position == 1 and close[i] < close[i-1])
        # Short exit: price breaks above weekly EMA50 or loss of momentum
        short_exit = (close[i] > ema50_1w_aligned[i]) or (position == -1 and close[i] > close[i-1])
        
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

name = "12h_WeeklyEMA50_Momentum_Breakout_Volume"
timeframe = "12h"
leverage = 1.0