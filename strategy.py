#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses weekly EMA for trend filter and weekly Donchian for breakout levels, aligned to 1d.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 15-25 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w Donchian and EMA to 1d
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, low_min_20)
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need 20-period Donchian (1w) + EMA50 (1w) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema50_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_1d[i]
        price_below_ema = close[i] < ema50_1d[i]
        
        # Price relative to 1w Donchian channels
        price_above_high = close[i] > donchian_high_1d[i]
        price_below_low = close[i] < donchian_low_1d[i]
        
        if position == 0:
            # Long: Price breaks above 1w Donchian high with volume and above 1w EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1w Donchian low with volume and below 1w EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 1w Donchian low OR below 1w EMA50
            if (close[i] < donchian_low_1d[i]) or (close[i] < ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 1w Donchian high OR above 1w EMA50
            if (close[i] > donchian_high_1d[i]) or (close[i] > ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0