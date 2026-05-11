#!/usr/bin/env python3
name = "6h_ADX_Trend_Filter_EMA_Crossover_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 12h EMA20 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    ema20_12h = pd.Series(df_12h['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate 12h ADX(14) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # Initial smoothed value
            result[period-1] = np.nansum(data[:period])
            # Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6-period and 18-period EMA for crossover
    ema6 = pd.Series(close).ewm(span=6, min_periods=6, adjust=False).mean().values
    ema18 = pd.Series(close).ewm(span=18, min_periods=18, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema6[i]) or np.isnan(ema18[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema12h = close[i] > ema20_12h_aligned[i]
        price_below_ema12h = close[i] < ema20_12h_aligned[i]
        ema_cross_up = ema6[i] > ema18[i]
        ema_cross_down = ema6[i] < ema18[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: EMA crossover up + above 12h EMA20 + strong trend
            if ema_cross_up and price_above_ema12h and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: EMA crossover down + below 12h EMA20 + strong trend
            elif ema_cross_down and price_below_ema12h and strong_trend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: EMA crossover down OR trend weakens
                if ema_cross_down or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: EMA crossover up OR trend weakens
                if ema_cross_up or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals