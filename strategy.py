#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter, volume confirmation, and ATR stoploss
# Long when price breaks above 4h Donchian high(20) + price > 12h EMA(50) + volume > 2x average
# Short when price breaks below 4h Donchian low(20) + price < 12h EMA(50) + volume > 2x average
# Exit when price returns to Donchian midline or opposite breakout occurs
# Uses 12h trend to filter breakouts in strong trends, targeting 100-180 trades over 4 years

name = "4h_donchian20_12h_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to midline OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midline OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: price breaks Donchian band + trend filter + volume
            if close[i] > high_roll[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout above upper band with trend and volume confirmation
                signals[i] = 0.25
                position = 1
            elif close[i] < low_roll[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout below lower band with trend and volume confirmation
                signals[i] = -0.25
                position = -1
    
    return signals