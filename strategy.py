#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
    # Camarilla levels provide precise intraday support/resistance.
    # 1d EMA filter ensures we trade with the daily trend.
    # Volume spike confirms breakout validity.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior day)
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.5 * (H - L)
    # L4 = Pivot - 1.5 * (H - L)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    camarilla_h4 = pivot_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = pivot_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below 1d EMA(20)
        uptrend = close[i] > ema_20_1d_aligned[i]
        downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Breakout conditions: price breaks Camarilla H4/L4 levels with volume and trend confirmation
        long_breakout = (close[i] > camarilla_h4_aligned[i]) and volume_filter and uptrend
        short_breakout = (close[i] < camarilla_l4_aligned[i]) and volume_filter and downtrend
        
        # Exit conditions: price returns to prior 12h bar's close (mean reversion)
        long_exit = close[i] < close[i-1]
        short_exit = close[i] > close[i-1]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0