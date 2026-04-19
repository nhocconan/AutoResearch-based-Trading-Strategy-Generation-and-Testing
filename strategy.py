#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation.
# Donchian channels identify price extremes and breakouts. EMA200 on daily provides long-term trend filter.
# Volume confirmation ensures breakouts are backed by participation. Works in bull/bear markets by
# avoiding counter-trend breakouts. Target: 20-50 trades/year per symbol.
name = "4h_Donchian20_EMA200_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian Channel (20) on 4h
    donch_period = 20
    upper = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align 1d EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 200, 20)  # Ensure Donchian, EMA200, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = upper[i]
        lower_band = lower[i]
        ema_200_val = ema_200_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper_band  # Price breaks above upper band
        bearish_breakout = price < lower_band  # Price breaks below lower band
        
        if position == 0:
            # Look for breakout in direction of daily trend
            if bullish_breakout and (price > ema_200_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and (price < ema_200_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to Donchian midpoint (mean reversion)
            midpoint = (upper_band + lower_band) / 2
            if price < midpoint:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to Donchian midpoint
            midpoint = (upper_band + lower_band) / 2
            if price > midpoint:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals