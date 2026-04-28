#/usr/bin/env python3
"""
6h_ADX_Trend_Strength_EMA_Crossover
Hypothesis: Uses 6h ADX to identify strong trending regimes (ADX > 25) and trades EMA crossovers (EMA12/EMA26) in the direction of the trend. Works in both bull and bear markets by only trading when trend strength is confirmed, avoiding whipsaws in ranging markets. Targets 15-35 trades/year by requiring strong trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for ADX calculation (trend strength filter)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 6h data
    period = 14
    # True Range
    tr1 = df_6h['high'] - df_6h['low']
    tr2 = abs(df_6h['high'] - df_6h['close'].shift(1))
    tr3 = abs(df_6h['low'] - df_6h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((df_6h['high'] - df_6h['high'].shift(1)) > (df_6h['low'].shift(1) - df_6h['low']), 
                                 np.maximum(df_6h['high'] - df_6h['high'].shift(1), 0), 0))
    dm_minus = pd.Series(np.where((df_6h['low'].shift(1) - df_6h['low']) > (df_6h['high'] - df_6h['high'].shift(1)), 
                                  np.maximum(df_6h['low'].shift(1) - df_6h['low'], 0), 0))
    
    # Smoothed values
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/period, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx_values = adx.values
    
    # Strong trend filter: ADX > 25
    strong_trend = adx_values > 25
    
    # Get 1d data for EMA calculation (entry signals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMAs on 1d data
    ema_12 = pd.Series(df_1d['close']).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(df_1d['close']).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align all higher timeframe data to 6h
    strong_trend_aligned = align_htf_to_ltf(prices, df_6h, strong_trend)
    ema_12_aligned = align_htf_to_ltf(prices, df_1d, ema_12)
    ema_26_aligned = align_htf_to_ltf(prices, df_1d, ema_26)
    
    # EMA crossover signals
    ema_cross_up = ema_12_aligned > ema_26_aligned  # Bullish crossover
    ema_cross_down = ema_12_aligned < ema_26_aligned  # Bearish crossover
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(strong_trend_aligned[i]) or 
            np.isnan(ema_12_aligned[i]) or 
            np.isnan(ema_26_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: strong trend + EMA crossover
        # Long: strong trend + bullish EMA crossover
        long_entry = strong_trend_aligned[i] and ema_cross_up[i]
        
        # Short: strong trend + bearish EMA crossover
        short_entry = strong_trend_aligned[i] and ema_cross_down[i]
        
        # Exit when trend weakens or opposite crossover
        long_exit = not strong_trend_aligned[i] or ema_cross_down[i]
        short_exit = not strong_trend_aligned[i] or ema_cross_up[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0  # Exit to flat
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

name = "6h_ADX_Trend_Strength_EMA_Crossover"
timeframe = "6h"
leverage = 1.0