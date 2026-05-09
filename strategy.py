#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day RSI and 4-hour Donchian channel breakout.
# In oversold conditions (RSI < 30), price tends to rebound; in overbought (RSI > 70), price tends to decline.
# Enters long when RSI(1d) < 30 and price breaks above 4h Donchian upper band.
# Enters short when RSI(1d) > 70 and price breaks below 4h Donchian lower band.
# Exits when RSI returns to neutral range (40-60) or opposite breakout occurs.
# Uses tight entry conditions to limit trades to 50-150 total over 4 years.
# Position size: 0.25.

name = "12h_RSI_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4-hour Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    donchian_upper = high_4h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_4h.rolling(window=20, min_periods=20).min()
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_values)
    
    # RSI conditions: oversold (<30) for long, overbought (>70) for short
    rsi_oversold = rsi_values < 30
    rsi_overbought = rsi_values > 70
    rsi_neutral = (rsi_values >= 40) & (rsi_values <= 60)
    
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold)
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought)
    rsi_neutral_aligned = align_htf_to_ltf(prices, df_1d, rsi_neutral)
    
    # Breakout conditions
    price_above_donchian_upper = close > donchian_upper_aligned
    price_below_donchian_lower = close < donchian_lower_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_oversold_aligned[i]) or np.isnan(rsi_overbought_aligned[i]) or
            np.isnan(rsi_neutral_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or np.isnan(price_above_donchian_upper[i]) or
            np.isnan(price_below_donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold + price breaks above 4h Donchian upper
            if rsi_oversold_aligned[i] and price_above_donchian_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought + price breaks below 4h Donchian lower
            elif rsi_overbought_aligned[i] and price_below_donchian_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR price breaks below 4h Donchian lower (reverse signal)
            if rsi_neutral_aligned[i] or price_below_donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR price breaks above 4h Donchian upper (reverse signal)
            if rsi_neutral_aligned[i] or price_above_donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals