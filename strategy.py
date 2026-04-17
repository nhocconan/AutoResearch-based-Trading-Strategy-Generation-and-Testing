#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1d Donchian channels (upper/lower band from past 20 days) to capture breakouts.
# Enters long when price breaks above upper band with volume and above 1w EMA50.
# Enters short when price breaks below lower band with volume and below 1w EMA50.
# Designed for low turnover (target: 7-25 trades/year) and robustness across bull/bear markets.
# In bull markets: captures momentum breakouts. In bear markets: filters out false breaks via 1w trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-day high/low)
    # Upper band = max(high of past 20 days)
    # Lower band = min(low of past 20 days)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and 1w indicators to 1d timeframe (same timeframe, but for consistency)
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 2.0 * 20-day average (strict to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for Donchian (20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_1d[i]) or 
            np.isnan(donchian_lower_1d[i]) or 
            np.isnan(ema50_1w_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_1w_1d[i]
        price_below_ema = close[i] < ema50_1w_1d[i]
        
        # Price relative to Donchian bands
        price_above_upper = close[i] > donchian_upper_1d[i]
        price_below_lower = close[i] < donchian_lower_1d[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume and above 1w EMA50
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume and below 1w EMA50
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower band OR below 1w EMA50
            if (close[i] < donchian_lower_1d[i]) or (close[i] < ema50_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper band OR above 1w EMA50
            if (close[i] > donchian_upper_1d[i]) or (close[i] > ema50_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0