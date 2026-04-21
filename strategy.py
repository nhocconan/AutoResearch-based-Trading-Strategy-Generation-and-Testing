#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above 6h Donchian high(20), price > 1d EMA50, and 1d volume > 1.5x 20-day average
# Short when price breaks below 6h Donchian low(20), price < 1d EMA50, and 1d volume > 1.5x 20-day average
# Exit when price crosses back through the opposite Donchian boundary or 1d EMA50
# Donchian channels provide clear breakout levels, EMA50 filters trend direction
# Volume confirms conviction, reducing false breakouts
# Target: 20-35 trades/year by requiring trend alignment + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Vectorized Donchian calculation
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get current 1d volume (4 6h bars per day)
        day_index = i // 4
        if day_index >= len(df_1d):
            volume = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0
        else:
            volume = df_1d['volume'].iloc[day_index]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian high, price > EMA50, volume confirmation
            if price > donch_high_val and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, price < EMA50, volume confirmation
            elif price < donch_low_val and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below Donchian low or crosses below EMA50
                if price < donch_low_val or price < ema50_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above Donchian high or crosses above EMA50
                if price > donch_high_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0