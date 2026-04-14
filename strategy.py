#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with volume confirmation and ADX trend filter.
# Long when price closes below lower BB(20,2) AND weekly trend is up (price > weekly EMA50) AND volume > 1.5x average.
# Short when price closes above upper BB(20,2) AND weekly trend is down (price < weekly EMA50) AND volume > 1.5x average.
# Exit when price returns to middle BB or weekly trend reverses.
# Bollinger Bands capture volatility expansion/contraction, weekly EMA50 filters trend direction.
# Volume confirmation ensures institutional participation. Designed for low frequency to avoid fee drag.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20
    
    # Calculate 20-day average volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need weekly EMA50 and BB20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below lower BB, weekly uptrend, high volume
            if (close[i] < lower_band[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                position = 1
                signals[i] = position_size
            # Short: price above upper BB, weekly downtrend, high volume
            elif (close[i] > upper_band[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or weekly trend turns down
            if (close[i] >= middle_band[i] or 
                close[i] < ema50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB or weekly trend turns up
            if (close[i] <= middle_band[i] or 
                close[i] > ema50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_BollingerBands_Volume_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0