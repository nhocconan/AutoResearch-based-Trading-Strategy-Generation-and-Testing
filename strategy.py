#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with 1-week EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + price > 1-week EMA200 + volume > 1.5x average
# Short when price breaks below Donchian(20) low + price < 1-week EMA200 + volume > 1.5x average
# Exit when price crosses Donchian midpoint or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 80-160 total trades over 4 years (20-40/year)
# Works in both bull and bear markets by following the higher timeframe trend

name = "6h_donchian20_1w_ema_vol_v1"
timeframe = "6h"
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
    
    # 1-week data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1-week EMA200
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema200_1w = close_1w_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Donchian(20) channels
    high_pd = pd.Series(high)
    low_pd = pd.Series(low)
    donch_high = high_pd.rolling(window=20, min_periods=20).max().values
    donch_low = low_pd.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Average volume for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 6-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses
            elif close[i] < donch_mid[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses
            elif close[i] > donch_mid[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume confirmation
            # Volume filter: volume > 1.5x average
            volume_confirmed = volume[i] > 1.5 * avg_volume[i]
            
            # Long: price breaks above Donchian high + above 1-week EMA200 + volume
            if close[i] > donch_high[i] and close[i] > ema200_1w_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + below 1-week EMA200 + volume
            elif close[i] < donch_low[i] and close[i] < ema200_1w_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals