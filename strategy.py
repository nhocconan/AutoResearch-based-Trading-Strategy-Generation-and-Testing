#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 4h for directional bias and entry timing, 1d for trend filter. Volume spike confirms breakout.
# Target: 15-37 trades/year (60-150 over 4 years) for 1h timeframe.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 (20%) to manage drawdown in volatile markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (directional bias)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Donchian and 1d EMA to 1h
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, low_min_20)
    ema50_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute for efficiency)
    session_mask = (prices.index.hour >= 8) & (prices.index.hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian (4h) + EMA50 (1d) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1h[i]) or 
            np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema50_1h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (balanced to avoid overtrading)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to 4h Donchian channels
        price_above_high = close[i] > donchian_high_1h[i]
        price_below_low = close[i] < donchian_low_1h[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_1h[i]
        price_below_ema = close[i] < ema50_1h[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume and above 1d EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low with volume and below 1d EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian low OR below 1d EMA50
            if (close[i] < donchian_low_1h[i]) or (close[i] < ema50_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian high OR above 1d EMA50
            if (close[i] > donchian_high_1h[i]) or (close[i] > ema50_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0