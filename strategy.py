#!/usr/bin/env python3
# Hypothesis: 1d ADX filter + 1w Supertrend trend filter with 1d RSI mean-reversion entries
# Long when: ADX(14) < 25 (ranging market), Supertrend(1w) bullish, RSI(14) < 30
# Short when: ADX(14) < 25 (ranging market), Supertrend(1w) bearish, RSI(14) > 70
# Exit when: RSI crosses back above 50 (long) or below 50 (short) OR ADX > 30 (trending)
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 10-25 trades/year.
# Designed for mean-reversion in ranging markets with trend filter to avoid whipsaws.

name = "1d_ADX_RSI_Supertrend_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Basic Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Bands
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    
    for i in range(1, len(close)):
        if close[i-1] <= final_ub[i-1]:
            final_ub[i] = min(final_ub[i], final_ub[i-1])
        else:
            final_ub[i] = basic_ub[i]
            
        if close[i-1] >= final_lb[i-1]:
            final_lb[i] = max(final_lb[i], final_lb[i-1])
        else:
            final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend = np.zeros(len(close))
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_ub[i]
        elif supertrend[i-1] == final_ub[i-1]:
            if close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Direction: 1 for uptrend, -1 for downtrend
    direction = np.where(close > supertrend, 1, -1)
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Supertrend for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    supertrend_1w, supertrend_dir_1w = calculate_supertrend(high_1w, low_1w, close_1w, period=10, multiplier=3.0)
    supertrend_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir_1w)
    
    # Get daily data for ADX and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) for ranging market filter
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # RSI(14) for mean-reversion entries
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(supertrend_dir_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        supertrend_dir = supertrend_dir_1w_aligned[i]
        
        if position == 0:
            # Enter long: ranging market (ADX < 25), bullish trend (Supertrend up), RSI oversold (< 30)
            if (adx_val < 25 and supertrend_dir == 1 and rsi_val < 30):
                signals[i] = 0.25
                position = 1
            # Enter short: ranging market (ADX < 25), bearish trend (Supertrend down), RSI overbought (> 70)
            elif (adx_val < 25 and supertrend_dir == -1 and rsi_val > 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 OR ADX > 30 (trending)
            if (rsi_val > 50) or (adx_val > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR ADX > 30 (trending)
            if (rsi_val < 50) or (adx_val > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals