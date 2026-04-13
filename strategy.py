#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
    # Long: price breaks above 20-period Donchian high AND weekly EMA200 uptrend AND volume > 1.3x average
    # Short: price breaks below 20-period Donchian low AND weekly EMA200 downtrend AND volume > 1.3x average
    # Exit: price returns to 10-period Donchian midpoint or volume drops below average
    # Using 1d for signal generation (Donchian breakout), 1w for trend filter
    # Discrete position sizing (0.25) to balance return and drawdown
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (pd.Series(high_1d).rolling(window=10, min_periods=10).max().values + 
                       pd.Series(low_1d).rolling(window=10, min_periods=10).min().values) / 2
    
    # Align 1d Donchian levels to 1d timeframe (no shift needed as we're already on 1d)
    donchian_high_20_aligned = donchian_high_20  # Already aligned to 1d
    donchian_low_20_aligned = donchian_low_20
    donchian_mid_10_aligned = donchian_mid_10
    
    # Get 1w data for trend filter (weekly EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        # Fallback: use 1d EMA50 if 1w not enough data
        ema_trend = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_trend_aligned = ema_trend
    else:
        close_1w = df_1w['close'].values
        ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
        ema_trend_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(donchian_mid_10_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long in weekly uptrend, only short in weekly downtrend
        long_trend_ok = close[i] > ema_trend_aligned[i]
        short_trend_ok = close[i] < ema_trend_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > donchian_high_20_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donchian_low_20_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to 10-period midpoint or volume dry-up
        long_exit = (close[i] < donchian_mid_10_aligned[i]) or not vol_confirm
        short_exit = (close[i] > donchian_mid_10_aligned[i]) or not vol_confirm
        
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

name = "1d_1w_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0