#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_40726
Hypothesis: Uses daily Camarilla levels (H4/L4) on 4h timeframe with volume confirmation and ADX regime filter.
Enters long when 4h close > daily H4 and volume > 1.5x 20-period volume average and ADX > 20.
Enters short when 4h close < daily L4 and volume > 1.5x 20-period volume average and ADX > 20.
Exits when price returns to prior 4h close or ADX < 15 (trend weakening).
Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
Works in both bull and bear markets by requiring volume expansion on breakouts and trend presence via ADX.
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
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous daily bar
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.125 * hl_range
    L4 = close_1d - 1.125 * hl_range
    
    # Calculate 20-period volume average on daily
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period-1] = np.mean(tr[1:period+1]) if period+1 <= len(tr) else np.mean(tr[1:])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period)
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x daily volume MA
        volume_expansion = volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # Regime filter: ADX > 20 for trend presence
        strong_trend = adx_1d_aligned[i] > 20
        weak_trend = adx_1d_aligned[i] < 15
        
        # Entry conditions: price CLOSES beyond H4/L4 with volume expansion and strong trend
        long_entry = (close[i] > H4_aligned[i]) and volume_expansion and strong_trend
        short_entry = (close[i] < L4_aligned[i]) and volume_expansion and strong_trend
        
        # Exit conditions: return to prior 4h close OR trend weakening
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        exit_long = position == 1 and (close[i] <= prev_close[i] or weak_trend)
        exit_short = position == -1 and (close[i] >= prev_close[i] or weak_trend)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "4h_1d_Camarilla_Breakout_40726"
timeframe = "4h"
leverage = 1.0