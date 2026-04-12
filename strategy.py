#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
    # Donchian(20) breakout captures strong momentum moves
    # 1w EMA50 trend filter ensures we only trade in direction of higher timeframe trend
    # Volume confirmation (1.5x 20-day average) filters weak breakouts
    # Works in bull/bear by only taking breakouts aligned with weekly trend
    # Target: 15-25 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h = np.full(len(df_1d), np.nan)
    donchian_l = np.full(len(df_1d), np.nan)
    
    for i in range(19, len(df_1d)):
        donchian_h[i] = np.max(high_1d[i-19:i+1])
        donchian_l[i] = np.min(low_1d[i-19:i+1])
    
    # Align 1d Donchian levels to 1d timeframe (identity alignment)
    donchian_h_aligned = donchian_h  # Already on 1d timeframe
    donchian_l_aligned = donchian_l  # Already on 1d timeframe
    
    # 1d volume spike filter (current volume > 1.5 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = np.full(len(df_1w), np.nan)
    multiplier = 2 / (50 + 1)
    for i in range(len(df_1w)):
        if i == 0:
            ema_50_1w[i] = close_1w[i]
        elif np.isnan(close_1w[i]):
            ema_50_1w[i] = ema_50_1w[i-1]
        else:
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align volume MA to 1d timeframe
    vol_ma_20_1d_aligned = vol_ma_20_1d  # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback period
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_h_aligned[i]
        short_breakout = close[i] < donchian_l_aligned[i]
        
        # Entry logic: breakout in direction of weekly trend with volume confirmation
        long_entry = long_breakout and uptrend and volume_spike
        short_entry = short_breakout and downtrend and volume_spike
        
        # Exit logic: opposite Donchian touch or volume dropout
        long_exit = close[i] < donchian_l_aligned[i] or (not volume_spike)
        short_exit = close[i] > donchian_h_aligned[i] or (not volume_spike)
        
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

name = "1d_1w_donchian_breakout_vol_trend_v1"
timeframe = "1d"
leverage = 1.0