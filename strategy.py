#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h upper band AND above 1d EMA50 AND volume > 2x average.
# Short when price breaks below 12h lower band AND below 1d EMA50 AND volume > 2x average.
# Exit when price crosses 12h EMA20 or volume drops below average.
# Uses 1d EMA50 for trend alignment to avoid counter-trend trades and reduce whipsaw.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel on 12h data (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA20 for exit condition
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need Donchian and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_20_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_20[i]
        lower_band = low_20[i]
        ema_trend = ema_50_aligned[i]
        ema_exit = ema_20_12h[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band AND above 1d EMA50 AND volume confirmed
            if price > upper_band and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band AND below 1d EMA50 AND volume confirmed
            elif price < lower_band and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below EMA20 or volume drops below average
            if price < ema_exit or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above EMA20 or volume drops below average
            if price > ema_exit or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals