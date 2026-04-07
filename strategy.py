#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter
# Long when price breaks above Donchian(20) high and 12h EMA(20) is rising
# Short when price breaks below Donchian(20) low and 12h EMA(20) is falling
# Volume confirmation filter: current volume > 20-period average
# Works in both bull and bear markets by following institutional breakouts
# Low frequency design targets 20-50 trades per year to minimize fee drag

name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 12h close for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian(20) on 4h data
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: EMA slope (current vs previous)
        if i >= 21:
            ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]
            ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        # Price levels
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (trend reversal)
            if close[i] < donch_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (trend reversal)
            if close[i] > donch_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long: break above Donchian high with volume and rising 12h EMA
            if close[i] > donch_high and vol_confirm and ema_rising:
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian low with volume and falling 12h EMA
            elif close[i] < donch_low and vol_confirm and ema_falling:
                position = -1
                signals[i] = -0.25
    
    return signals