#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1w ADX regime filter
    # Long: price breaks above 20-period 12h Donchian high + 1d volume > 1.5x 20-period average + 1w ADX > 25
    # Short: price breaks below 20-period 12h Donchian low + 1d volume > 1.5x 20-period average + 1w ADX > 25
    # Range regime (1w ADX < 20): fade at Donchian extremes with volume confirmation
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(period-1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper_12h, donchian_lower_12h = calculate_donchian(high_12h, low_12h, 20)
    donchian_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
        
        # Wilder's smoothing
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_12h_aligned[i]) or np.isnan(donchian_lower_12h_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime
        strong_trend = adx_1w_aligned[i] > 25
        ranging = adx_1w_aligned[i] < 20
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * volume_ma_1d_aligned[i]
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if strong_trend:
            # Strong trend: breakout in direction of trend
            long_entry = close[i] > donchian_upper_12h_aligned[i] and volume_confirm
            short_entry = close[i] < donchian_lower_12h_aligned[i] and volume_confirm
        elif ranging:
            # Range regime: mean reversion at extremes
            long_entry = close[i] < donchian_lower_12h_aligned[i] and volume_confirm
            short_entry = close[i] > donchian_upper_12h_aligned[i] and volume_confirm
        
        # Exit logic: opposite signal or regime change
        long_exit = False
        short_exit = False
        
        if strong_trend:
            long_exit = close[i] < donchian_lower_12h_aligned[i]
            short_exit = close[i] > donchian_upper_12h_aligned[i]
        elif ranging:
            long_exit = close[i] > donchian_upper_12h_aligned[i]
            short_exit = close[i] < donchian_lower_12h_aligned[i]
        
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

name = "12h_1d_1w_donchian_volume_adx_regime_v1"
timeframe = "12h"
leverage = 1.0