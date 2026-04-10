#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# - Donchian breakout: price breaks above 20-period high (long) or below 20-period low (short)
# - 12h EMA50 trend filter: ensures we trade with higher timeframe trend
# - Volume confirmation: current volume > 1.5x 20-period average to avoid false breakouts
# - Exit: Donchian opposite breakout
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to avoid fee drag
# - Position size: 0.25 (25% of capital) for balanced risk/return

name = "4h_12h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous period high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous period low
        
        # Exit conditions: opposite Donchian breakout
        exit_long = close[i] < lowest_low[i-1]   # Price breaks below Donchian low
        exit_short = close[i] > highest_high[i-1] # Price breaks above Donchian high
        
        # 12h trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        close_12h_current = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_current)
        bullish_trend = not np.isnan(close_12h_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_12h_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_12h_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_12h_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian breakout up AND bullish trend AND volume confirmation
            if long_breakout and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakout down AND bearish trend AND volume confirmation
            elif short_breakout and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: opposite Donchian breakout
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals