#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 Trend Filter + Volume Spike
# Long when price breaks above Donchian high(20) on 1d, price > 1w EMA34 (uptrend), and volume > 1.5x 20-day average
# Short when price breaks below Donchian low(20) on 1d, price < 1w EMA34 (downtrend), and volume > 1.5x 20-day average
# Donchian channels provide clear breakout levels; 1w EMA34 filters for higher timeframe trend; volume confirms conviction
# Target: 10-25 trades/year by requiring 1d breakout + 1w trend + volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (1d values available after daily close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute volume moving average (20-period on lower timeframe)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs 1w EMA34
        uptrend = price > ema34_1w_aligned[i]
        downtrend = price < ema34_1w_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high(20) in uptrend
                if price > donchian_high_aligned[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low(20) in downtrend
                elif price < donchian_low_aligned[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below Donchian low(20) or trend fails
                if price < donchian_low_aligned[i] or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above Donchian high(20) or trend fails
                if price > donchian_high_aligned[i] or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0