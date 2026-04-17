#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly EMA trend filter and volume spike.
# Uses weekly Donchian channels (20-week) for breakout levels and weekly EMA50 for trend filter.
# Volume confirmation (current volume > 2x 20-period average) ensures breakout strength.
# Designed for low turnover to capture major trends while minimizing fee drag.
# Target: 10-25 trades/year to stay within optimal range for 1d timeframe.
# Works in both bull and bear markets by following the trend with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly Donchian and EMA to daily
    donchian_high_daily = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_daily = align_htf_to_ltf(prices, df_1w, low_min_20)
    ema50_daily = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian (1w) + EMA50 (1w) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_daily[i]) or 
            np.isnan(donchian_low_daily[i]) or 
            np.isnan(ema50_daily[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_daily[i]
        price_below_ema = close[i] < ema50_daily[i]
        
        # Price relative to weekly Donchian channels
        price_above_high = close[i] > donchian_high_daily[i]
        price_below_low = close[i] < donchian_low_daily[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume and above weekly EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume and below weekly EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian low OR below weekly EMA50
            if (close[i] < donchian_low_daily[i]) or (close[i] < ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian high OR above weekly EMA50
            if (close[i] > donchian_high_daily[i]) or (close[i] > ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0