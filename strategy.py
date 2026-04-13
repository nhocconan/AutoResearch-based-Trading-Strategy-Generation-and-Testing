#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
    # Long: price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND price > 1d EMA200
    # Short: price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND price < 1d EMA200
    # Exit: price returns to Donchian(20) midpoint (mean reversion in 12h timeframe)
    # Using 1d for EMA200 (trend filter) and Donchian calculation (structure), 12h only for entry timing
    # Discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter and Donchian levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d Donchian(20) levels (based on previous 20 daily bars)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high = max(high over last 20 days)
    donchian_high_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-20:i])
    
    # Donchian low = min(low over last 20 days)
    donchian_low_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donchian_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Donchian midpoint = (high + low) / 2
    donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2
    
    # Align 1d indicators to 12h (wait for completed 1d bar)
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    donchian_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(donchian_mid_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA200, only short if price < 1d EMA200
        long_trend_ok = close[i] > ema_200_1d_aligned[i]
        short_trend_ok = close[i] < ema_200_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > donchian_high_1d_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donchian_low_1d_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to Donchian midpoint (mean reversion)
        long_exit = close[i] < donchian_mid_1d_aligned[i]
        short_exit = close[i] > donchian_mid_1d_aligned[i]
        
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

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0