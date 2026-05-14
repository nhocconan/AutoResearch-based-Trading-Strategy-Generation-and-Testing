#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h ATR-based breakout with 12h trend filter and volume confirmation.
    # ATR breakout captures volatility expansion after consolidation.
    # 12h EMA filter ensures we trade with the higher timeframe trend.
    # Volume spike confirms breakout validity.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ATR calculation and EMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(14)
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below 12h EMA(20)
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        # Calculate 6h ATR-based breakout levels using prior bar's ATR
        atr_val = atr_14_aligned[i-1]
        upper_breakout = close[i-1] + 0.5 * atr_val
        lower_breakout = close[i-1] - 0.5 * atr_val
        
        # Breakout conditions: price breaks ATR bands with volume and trend confirmation
        long_breakout = (close[i] > upper_breakout) and volume_filter and uptrend
        short_breakout = (close[i] < lower_breakout) and volume_filter and downtrend
        
        # Exit conditions: price returns to prior bar's close (mean reversion)
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

name = "6h_12h_atr_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0