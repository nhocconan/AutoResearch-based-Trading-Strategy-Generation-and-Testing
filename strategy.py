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
    
    # Get 12h data for higher timeframe trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h ATR for volatility filter
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h moving average of ATR for volatility regime
    atr_ma_20_12h = pd.Series(atr_12h_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA34, pivots, volume MA20, ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema34_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(atr_ma_20_12h[i]) or
            np.isnan(r1_4h[i]) or
            np.isnan(s1_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: 12h ATR > 12h ATR MA20 (avoid low volatility regimes)
        volatility_filter = atr_12h_aligned[i] > atr_ma_20_12h[i]
        # 12h trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema34_12h_aligned[i]
        trend_down = close[i] < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, volatility AND 12h uptrend
            if (close[i] > r1_4h[i] and volume_filter and volatility_filter and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, volatility AND 12h downtrend
            elif (close[i] < s1_4h[i] and volume_filter and volatility_filter and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 12h EMA34 or volatility drops
            if close[i] < ema34_12h_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 12h EMA34 or volatility drops
            if close[i] > ema34_12h_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34_PivotBreakout_Volume"
timeframe = "4h"
leverage = 1.0