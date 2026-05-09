#!/usr/bin/env python3
name = "1D_1W_Triple_Confirmation"
timeframe = "1d"
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
    
    # Get weekly data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    if len(close_1w) >= 50:
        ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema50_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) channel on daily
    donchian_high = np.full_like(close_1d, np.nan)
    donchian_low = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 20:
            donchian_high[i] = np.max(high_1d[i-20:i])
            donchian_low[i] = np.min(low_1d[i-20:i])
        else:
            donchian_high[i] = np.max(high_1d[:i+1]) if i > 0 else high_1d[i]
            donchian_low[i] = np.min(low_1d[:i+1]) if i > 0 else low_1d[i]
    
    # Calculate RSI(14) on daily
    rsi = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_1d)
        avg_loss = np.zeros_like(close_1d)
        
        # First average
        if len(gain) >= 14:
            avg_gain[13] = np.mean(gain[:14])
            avg_loss[13] = np.mean(loss[:14])
            
            # Wilder smoothing
            for i in range(14, len(close_1d)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        # Calculate RSI
        for i in range(13, len(close_1d)):
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    # Align daily indicators to main timeframe (they're already daily, so just forward fill)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, 20, 14)  # Need weekly EMA50, Donchian20, RSI14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        # Weekly trend: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Donchian breakout
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        if position == 0:
            # Enter long: Weekly uptrend + bullish breakout + RSI not overbought
            if weekly_uptrend and breakout_up and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly downtrend + bearish breakout + RSI not oversold
            elif weekly_downtrend and breakout_down and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR bearish breakout
            if not weekly_uptrend or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR bullish breakout
            if not weekly_downtrend or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals