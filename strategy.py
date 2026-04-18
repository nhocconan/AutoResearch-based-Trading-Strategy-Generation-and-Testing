#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Trend_Filter_v2
Hypothesis: Weekly Donchian breakouts with daily trend filter capture strong momentum in both bull and bear markets.
Weekly highs/lows act as major support/resistance. Daily EMA filter ensures trades align with intermediate trend.
Designed for low trade frequency (10-20/year) to minimize fee drag on 1d timeframe.
"""

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
    
    # Get weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's levels only
    donchian_high_prev = np.roll(donchian_high, 1)
    donchian_low_prev = np.roll(donchian_low, 1)
    donchian_high_prev[0] = np.nan
    donchian_low_prev[0] = np.nan
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_prev)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_prev)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: avoid extremely low volatility
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_filter = volatility_filter[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with daily uptrend and sufficient volatility
            if price > upper and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with daily downtrend and sufficient volatility
            elif price < lower and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to weekly Donchian low or breaks below daily EMA
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to weekly Donchian high or breaks above daily EMA
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Trend_Filter_v2"
timeframe = "1d"
leverage = 1.0