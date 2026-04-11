#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate weekly Donchian Channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Highest high and lowest low over past 20 weekly bars
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # Calculate 12h ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility filter: ATR > 0.5 * 50-period ATR average (avoid low volatility chop)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma_50)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Breakout conditions
        long_breakout = price_high > highest_high_aligned[i]
        short_breakout = price_low < lowest_low_aligned[i]
        
        # Entry conditions with volatility and volume filters
        long_signal = long_breakout and volatility_filter[i] and volume_confirmed
        short_signal = short_breakout and volatility_filter[i] and volume_confirmed
        
        # Exit when price returns to the midpoint of the weekly Donchian channel
        midpoint = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2.0
        exit_long = position == 1 and price_close < midpoint
        exit_short = position == -1 and price_close > midpoint
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Donchian breakout strategy on 12h timeframe.
# Uses weekly Donchian Channel (20-period) to identify major support/resistance levels.
# Enters long when price breaks above weekly high with volatility filter (ATR > 50% of its MA) and volume confirmation (>1.3x avg volume).
# Enters short when price breaks below weekly low with same filters.
# Exits when price returns to the midpoint of the weekly Donchian channel.
# Volatility filter avoids low-volatility choppy markets where breakouts fail.
# Works in both bull and bear markets by trading breakouts in either direction.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Weekly timeframe provides structural support/resistance that holds across market regimes.