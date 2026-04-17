#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20) with 1w EMA34 trend filter and volume confirmation.
# Uses 1d Donchian channels (20) for breakout signals, filtered by 1w EMA34 trend and volume spikes.
# Enters long when price breaks above upper Donchian with volume and above 1w EMA34.
# Enters short when price breaks below lower Donchian with volume and below 1w EMA34.
# Designed to capture medium-term trends with low turnover (target: 7-25 trades/year).
# Works in bull markets (breakout momentum) and bear markets (via trend filter).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20) from previous day
    # Upper = max(high[last 20 days]), Lower = min(low[last 20 days])
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d and 1w indicators to 1d timeframe
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema34_1w_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need sufficient data for Donchian(20) and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_1d[i]) or 
            np.isnan(donchian_lower_1d[i]) or 
            np.isnan(ema34_1w_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA34
        price_above_ema = close[i] > ema34_1w_1d[i]
        price_below_ema = close[i] < ema34_1w_1d[i]
        
        # Price relative to Donchian levels
        price_above_upper = close[i] > donchian_upper_1d[i]
        price_below_lower = close[i] < donchian_lower_1d[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above 1w EMA34
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and below 1w EMA34
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below lower Donchian OR below 1w EMA34
            if (close[i] < donchian_lower_1d[i]) or (close[i] < ema34_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above upper Donchian OR above 1w EMA34
            if (close[i] > donchian_upper_1d[i]) or (close[i] > ema34_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0