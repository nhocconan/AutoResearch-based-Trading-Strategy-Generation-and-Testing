#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend and volume confirmation
# Uses Donchian(20) for breakouts, weekly EMA for trend, and volume spike for confirmation
# Designed to capture strong momentum moves in both bull and bear markets
# Target: 7-25 trades/year (30-100 total) for 1d timeframe
name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA for trend filter (21-period)
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate weekly volume SMA for volume context (10-period)
    vol_sma_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily Donchian channels (20-period)
    # Upper band: 20-period high
    # Lower band: 20-period low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w[i]) or np.isnan(volume_1w[i]) or 
            np.isnan(vol_sma_1w[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly values for current daily bar
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)[i]
        vol_sma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_1w)[i]
        
        # Trend filter: price above/below 21 EMA on weekly
        uptrend = close[i] > ema_1w_aligned
        downtrend = close[i] < ema_1w_aligned
        
        # Volume filter: current volume above 2.5x weekly average volume
        volume_filter = volume[i] > (vol_sma_1w_aligned * 2.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reversal
            if close[i] < donchian_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reversal
            if close[i] > donchian_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high + uptrend + volume filter
            if close[i] > donchian_high[i] and uptrend and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low + downtrend + volume filter
            elif close[i] < donchian_low[i] and downtrend and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals