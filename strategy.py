#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (weekly EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period EMA on weekly close
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channel on daily
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume on daily
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to minute timeframe (they're already aligned for 1d timeframe)
    # Since prices are 1d timeframe, no further alignment needed
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    rsi_14_aligned = rsi_14
    vol_ma_20_aligned = vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > donchian_high_aligned[i]
        donchian_breakdown_down = close[i] < donchian_low_aligned[i]
        
        # Volume confirmation: volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # RSI filter: avoid extreme levels
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Entry conditions
        long_entry = above_weekly_ema and donchian_breakout_up and volume_confirm and rsi_not_overbought
        short_entry = below_weekly_ema and donchian_breakdown_down and volume_confirm and rsi_not_oversold
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = position == 1 and (below_weekly_ema or rsi_14_aligned[i] > 80)
        exit_short = position == -1 and (above_weekly_ema or rsi_14_aligned[i] < 20)
        
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

name = "1d_weekly_ema_donchian_volume_rsi_filter"
timeframe = "1d"
leverage = 1.0