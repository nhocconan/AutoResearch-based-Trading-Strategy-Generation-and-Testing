#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves
# 1d EMA50 ensures we only trade in direction of higher timeframe trend
# Volume confirmation (2.0x 20-period average) filters false breakouts
# Works in bull/bear by only taking breakouts in direction of 1d EMA50 trend
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year)

name = "12h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 1d (using previous 20 completed days)
    # Upper band = highest high of previous 20 days
    # Lower band = lowest low of previous 20 days
    prev_high = df_1d['high'].shift(1).rolling(window=20, min_periods=20).max().values
    prev_low = df_1d['low'].shift(1).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (completed 1d bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Donchian upper with 1d uptrend (close > EMA50)
            long_breakout = close[i] > donchian_upper_aligned[i]
            # Short breakdown: price < Donchian lower with 1d downtrend (close < EMA50)
            short_breakout = close[i] < donchian_lower_aligned[i]
            
            # 1d EMA50 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_50_1d_aligned[i]
            ema_trend_down = close[i] < ema_50_1d_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower or trend reversal (close < EMA50)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper or trend reversal (close > EMA50)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals