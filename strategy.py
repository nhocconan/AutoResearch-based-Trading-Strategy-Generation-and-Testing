#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper channel with price above 1d EMA50 and volume spike (>2.0x average).
# Short when price breaks below lower channel with price below 1d EMA50 and volume spike.
# Uses daily EMA50 to filter trend direction, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 15-40 trades/year per symbol (~60-160 total over 4 years).
name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # Using rolling window on high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50, 20)  # Need volume MA, EMA, and Donchian data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_30[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_30[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper channel AND above 1d EMA50
            if price > upper and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel AND below 1d EMA50
            elif price < lower and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower channel or below 1d EMA50
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper channel or above 1d EMA50
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals