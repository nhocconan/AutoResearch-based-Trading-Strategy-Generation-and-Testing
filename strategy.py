#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Long when price breaks above 1d Donchian upper band, price > 1w EMA50, and 1d volume > 1.5x 20-day average
# Short when price breaks below 1d Donchian lower band, price < 1w EMA50, and 1d volume > 1.5x 20-day average
# Exit when price crosses back inside Donchian bands or crosses 1w EMA50
# Uses price channels for trend-following with volume and trend filters
# Target: 10-25 trades/year by requiring breakout + trend alignment + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get daily open, high, low, close for breakout detection
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Align OHLCV to daily timeframe
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        high = high_1d_aligned[i]
        low = low_1d_aligned[i]
        close = close_1d_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        volume = volume_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high, price > EMA50, volume confirmation
            if high > donch_high_val and close > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, price < EMA50, volume confirmation
            elif low < donch_low_val and close < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian low or crosses below EMA50
                if low < donch_low_val or close < ema50_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian high or crosses above EMA50
                if high > donch_high_val or close > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0