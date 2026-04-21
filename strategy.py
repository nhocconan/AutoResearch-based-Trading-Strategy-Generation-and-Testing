#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume spike
# Long when price breaks above Donchian high(20) AND price > 1d EMA50 AND 1d volume > 2x 20-day average
# Short when price breaks below Donchian low(20) AND price < 1d EMA50 AND 1d volume > 2x 20-day average
# Exit when price crosses 1d EMA50 (trend reversal) OR opposite Donchian breakout occurs
# Donchian channels provide clear breakout levels, EMA50 filters trend direction, volume spike confirms conviction
# Target: 20-40 trades/year by requiring volume spike + trend alignment + breakout confirmation

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
    
    # Align all 1d indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get 1d volume for current day (4 bars per day in 4h timeframe)
        day_index = i // 4
        if day_index >= len(df_1d):
            day_index = len(df_1d) - 1
        volume = df_1d['volume'].iloc[day_index] if day_index >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 2x 20-day average
        volume_confirm = volume > 2.0 * vol_ma if day_index >= 20 else False
        
        if position == 0:
            # Long: Price breaks above Donchian high, price > EMA50, volume confirmation
            if price > donch_high and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, price < EMA50, volume confirmation
            elif price < donch_low and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below EMA50 (trend reversal) OR breaks below Donchian low
                if price < ema50_val or price < donch_low:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above EMA50 (trend reversal) OR breaks above Donchian high
                if price > ema50_val or price > donch_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0