#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
    # Long: price > Donchian_high(20) AND EMA50 rising AND volume > 1.5x avg
    # Short: price < Donchian_low(20) AND EMA50 falling AND volume > 1.5x avg
    # Exit: opposite Donchian break or volume dry-up
    # Using 12h timeframe for low trade frequency, Donchian for structure,
    # 1d EMA50 for trend filter, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate EMA50 direction (1 = rising, -1 = falling, 0 = flat)
    ema_dir = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(ema_50_aligned[i]) and not np.isnan(ema_50_aligned[i-1]):
            if ema_50_aligned[i] > ema_50_aligned[i-1]:
                ema_dir[i] = 1
            elif ema_50_aligned[i] < ema_50_aligned[i-1]:
                ema_dir[i] = -1
            else:
                ema_dir[i] = ema_dir[i-1]
        else:
            ema_dir[i] = ema_dir[i-1] if i > 0 else 0
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        donch_high[i] = np.max(window_high)
        donch_low[i] = np.min(window_low)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_dir[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA50 rising = bullish bias, EMA50 falling = bearish bias
        bullish_bias = ema_dir[i] == 1
        bearish_bias = ema_dir[i] == -1
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian break + trend bias + volume confirmation
        long_entry = (close[i] > donch_high[i]) and bullish_bias and vol_confirm
        short_entry = (close[i] < donch_low[i]) and bearish_bias and vol_confirm
        
        # Exit logic: opposite Donchian break or volume dry-up
        long_exit = (close[i] < donch_low[i]) or not vol_confirm
        short_exit = (close[i] > donch_high[i]) or not vol_confirm
        
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

name = "12h_1d_donchian_ema_volume_v2"
timeframe = "12h"
leverage = 1.0