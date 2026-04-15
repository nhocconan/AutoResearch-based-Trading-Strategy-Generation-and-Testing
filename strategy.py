#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback with 4h/1d trend alignment and session filter
# Uses 4h EMA50 for trend direction, 1d ADX for trend strength filter, and 1h EMA21 for entry timing.
# Only takes pullbacks to EMA21 in the direction of the higher timeframe trend.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Load 1d data for trend strength (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Directional Indicators
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h EMA21 for entry timing
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(ema21[i])):
            continue
        
        # Check session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            continue
        
        # Long entry: pullback to EMA21 in uptrend (4h EMA50 up + 1d ADX > 25)
        if (ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and  # 4h trend up
            adx_aligned[i] > 25 and                         # Strong trend
            low[i] <= ema21[i] and                          # Pullback to EMA21
            close[i] > ema21[i] and                         # Close above EMA21 (confirm bounce)
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: pullback to EMA21 in downtrend (4h EMA50 down + 1d ADX > 25)
        elif (ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and  # 4h trend down
              adx_aligned[i] > 25 and                         # Strong trend
              high[i] >= ema21[i] and                         # Pullback to EMA21
              close[i] < ema21[i] and                         # Close below EMA21 (confirm bounce)
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend weakness or opposite signal
        elif position == 1 and (ema50_4h_aligned[i] < ema50_4h_aligned[i-1] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema50_4h_aligned[i] > ema50_4h_aligned[i-1] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_EMA21_Pullback_4hTrend_1dADX"
timeframe = "1h"
leverage = 1.0