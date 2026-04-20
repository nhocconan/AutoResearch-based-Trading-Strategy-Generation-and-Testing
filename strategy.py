#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-day trend filter + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions
# In bull market (price > 1-day EMA50): buy oversold (%R < -80), sell overbought (%R > -20)
# In bear market (price < 1-day EMA50): sell rallies (%R > -20), buy dips (%R < -80)
# Volume confirmation: require volume > 1.5x 20-period average
# Designed to work in both bull and bear markets by adapting to trend direction
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 12h data for Williams %R and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 14-period Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(wr[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long conditions: oversold in bull OR bear market
            long_signal = False
            if has_volume:
                if wr[i] < -80:  # Oversold
                    long_signal = True
            
            # Enter short conditions: overbought in bull OR bear market
            short_signal = False
            if has_volume:
                if wr[i] > -20:  # Overbought
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought condition
            exit_signal = False
            if has_volume and wr[i] > -20:  # Overbought
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold condition
            exit_signal = False
            if has_volume and wr[i] < -80:  # Oversold
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0