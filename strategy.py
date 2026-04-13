#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and ATR(14) volatility filter
    # Donchian breakouts capture momentum in trending markets, EMA200 avoids counter-trend trades
    # ATR filter ensures sufficient volatility for meaningful moves, reducing whipsaws
    # Target: 20-30 trades/year (80-120 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h ATR(14) for volatility filter
    atr_4h = np.full(len(high_4h), np.nan)
    tr_4h = np.zeros(len(high_4h))
    for i in range(1, len(high_4h)):
        tr_4h[i] = max(
            high_4h[i] - low_4h[i],
            abs(high_4h[i] - close_4h[i-1]),
            abs(low_4h[i] - close_4h[i-1])
        )
    for i in range(14, len(high_4h)):
        if i == 14:
            atr_4h[i] = np.mean(tr_4h[1:15])
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate 4h ATR moving average (20-period) for volatility regime filter
    atr_ma_4h = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        atr_ma_4h[i] = np.mean(atr_4h[i-20:i])
    
    # Align all indicators to LTF (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    atr_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(atr_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # 1d EMA200 trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # ATR volatility filter: current ATR > 1.5x ATR_MA (ensures sufficient volatility)
        volatility_filter = atr_4h_aligned[i] > (1.5 * atr_ma_4h_aligned[i])
        
        # Entry logic: Breakout + trend alignment + volatility filter
        long_entry = long_breakout and bullish_trend and volatility_filter
        short_entry = short_breakout and bearish_trend and volatility_filter
        
        # Exit logic: opposite Donchian breakout or trend reversal
        long_exit = (close[i] < donchian_low_aligned[i]) or not bullish_trend
        short_exit = (close[i] > donchian_high_aligned[i]) or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_ema200_atr_filter_v1"
timeframe = "4h"
leverage = 1.0